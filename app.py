from __future__ import annotations

from flask import Flask, request, render_template_string, jsonify
import json
import re
import sqlite3
from urllib.parse import urlparse
from datetime import datetime, timedelta
from typing import Any

app = Flask(__name__)

DB_NAME = "fake_news_history.db"


def get_db_connection(*, row_factory: bool = False) -> sqlite3.Connection:
    conn = sqlite3.connect(DB_NAME, timeout=10)
    if row_factory:
        conn.row_factory = sqlite3.Row
    return conn

# -----------------------------
# Database setup
# -----------------------------
def init_db() -> None:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout = 10000")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            headline TEXT,
            article_text TEXT,
            source_url TEXT,
            source_domain TEXT,
            predicted_label TEXT,
            risk_band TEXT,
            fake_score REAL,
            confidence REAL,
            explanation TEXT,
            red_flags_json TEXT,
            positive_signals_json TEXT,
            breakdown_json TEXT,
            created_at TEXT
        )
    """
    )

    # Lightweight schema migration for existing databases.
    existing_columns = {
        row[1] for row in cur.execute("PRAGMA table_info(analysis_history)").fetchall()
    }
    required = {
        "source_domain": "TEXT",
        "risk_band": "TEXT",
        "red_flags_json": "TEXT",
        "positive_signals_json": "TEXT",
        "breakdown_json": "TEXT",
    }
    for col, col_type in required.items():
        if col not in existing_columns:
            cur.execute(f"ALTER TABLE analysis_history ADD COLUMN {col} {col_type}")

    conn.commit()
    conn.close()


# -----------------------------
# Demo training data
# label: 1 = likely fake, 0 = likely real
# -----------------------------
TRAINING_DATA = [
    ("Government releases verified public health update with cited data from official agencies", 0),
    ("Breaking report confirms policy change with official documents and expert interviews", 0),
    ("University study published in peer reviewed journal shows measured results", 0),
    ("Local authorities provide detailed update after investigation and public statement", 0),
    ("Reuters confirms election results after multiple state certifications", 0),
    ("Scientists publish climate findings with methods, data, and transparent limitations", 0),
    ("Hospital system announces new safety procedures in official press release", 0),
    ("Court filing reveals timeline supported by public records and legal documents", 0),
    ("State audit releases budget findings with full methodology and evidence", 0),
    ("Independent watchdog confirms report using public datasets and records", 0),
    ("Energy regulator issues formal statement with technical appendix", 0),
    ("International agency publishes verified humanitarian situation update", 0),
    ("You will not believe this shocking secret they hid from everyone", 1),
    ("Miracle cure discovered overnight doctors hate it click now", 1),
    ("Secret elites exposed in unbelievable scandal with no mainstream coverage", 1),
    ("This one weird trick proves the entire news industry is lying", 1),
    ("Breaking celebrity clone conspiracy confirmed by anonymous insiders", 1),
    ("End of the world prediction is true share before it gets deleted", 1),
    ("Massive fraud uncovered with zero evidence but everyone is talking about it", 1),
    ("The truth they do not want you to know revealed in viral post", 1),
    ("Guaranteed proof government controlled weather machines exposed", 1),
    ("Mainstream media refuses to report this unbelievable cure", 1),
    ("Viral post claims secret microchips in food with no sources", 1),
    ("Hidden agenda uncovered wake up before they silence this", 1),
]

vectorizer = None
classifier = None
APP_READY = False
ML_READY = False


def train_demo_model() -> None:
    global ML_READY, vectorizer, classifier
    texts = [text for text, _ in TRAINING_DATA]
    labels = [label for _, label in TRAINING_DATA]
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
        classifier = LogisticRegression(max_iter=1200)
        X = vectorizer.fit_transform(texts)
        classifier.fit(X, labels)
        ML_READY = True
    except Exception:
        vectorizer = None
        classifier = None
        ML_READY = False


def ensure_app_ready() -> None:
    global APP_READY
    if APP_READY:
        return

    init_db()
    APP_READY = True


# -----------------------------
# Lexicons and thresholds
# -----------------------------
TRUSTED_DOMAINS = {
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "npr.org",
    "nytimes.com",
    "wsj.com",
    "theguardian.com",
    "cbsnews.com",
    "abcnews.go.com",
    "cnn.com",
    "who.int",
    "cdc.gov",
    "nih.gov",
    "un.org",
}

SUSPICIOUS_PHRASES = [
    "you will not believe",
    "they do not want you to know",
    "they don't want you to know",
    "one weird trick",
    "share before it gets deleted",
    "miracle cure",
    "hidden truth",
    "mainstream media won't tell you",
    "wake up",
    "guaranteed proof",
    "secret elites",
    "anonymous insiders",
    "100% proof",
    "shocking",
    "unbelievable",
]

PARTISAN_LOADED_WORDS = [
    "traitor",
    "corrupt",
    "evil",
    "radical",
    "enemy",
    "propaganda",
    "betrayal",
    "rigged",
    "brainwashed",
    "destroying",
]

EVIDENCE_WORDS = [
    "according to",
    "reported by",
    "study",
    "data",
    "source",
    "official",
    "document",
    "research",
    "agency",
    "court",
    "peer reviewed",
    "analysis",
    "statement",
]

ALL_CAPS_THRESHOLD = 0.33


# -----------------------------
# Utility functions
# -----------------------------
def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(value, high))


def extract_domain(url: str) -> str:
    if not url:
        return ""
    try:
        parsed = urlparse(url if url.startswith(("http://", "https://")) else f"https://{url}")
        return parsed.netloc.lower().replace("www.", "")
    except Exception:
        return ""


def looks_trusted(domain: str) -> bool:
    if not domain:
        return False
    return domain.endswith(".gov") or domain.endswith(".edu") or domain in TRUSTED_DOMAINS


def count_matches(text: str, phrases: list[str]) -> list[str]:
    lowered = text.lower()
    return [p for p in phrases if p in lowered]


def all_caps_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    caps = [ch for ch in letters if ch.isupper()]
    return len(caps) / len(letters)


def excessive_punctuation_score(text: str) -> float:
    drama = text.count("!") + text.count("?")
    return clamp(drama / 10.0)


def evidence_strength(text: str) -> float:
    lowered = text.lower()
    hits = sum(lowered.count(term) for term in EVIDENCE_WORDS)
    return clamp(hits / 8.0)


def emotional_bias_score(text: str) -> float:
    lowered = text.lower()
    hits = sum(lowered.count(word) for word in PARTISAN_LOADED_WORDS)
    return clamp(hits / 8.0)


def heuristic_ml_probability(text: str) -> float:
    lowered = text.lower()
    fake_hits = sum(lowered.count(term) for term in SUSPICIOUS_PHRASES + PARTISAN_LOADED_WORDS)
    real_hits = sum(lowered.count(term) for term in EVIDENCE_WORDS)
    length_bonus = 0.08 if len(lowered.split()) < 45 else 0.0
    score = 0.48 + min(fake_hits * 0.06, 0.32) - min(real_hits * 0.05, 0.28) + length_bonus
    return clamp(score, 0.05, 0.95)


def readability_risk(text: str) -> float:
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    words = re.findall(r"\b\w+\b", text)
    if not sentences or not words:
        return 0.5

    avg_words = len(words) / len(sentences)
    if avg_words < 7:
        return 0.45
    if avg_words > 35:
        return 0.40
    return 0.12


def get_ml_fake_probability(text: str) -> float:
    if not text.strip():
        return 0.5
    if not ML_READY:
        return heuristic_ml_probability(text)
    try:
        X = vectorizer.transform([text])
        return float(classifier.predict_proba(X)[0][1])
    except Exception:
        return heuristic_ml_probability(text)


def get_risk_band(fake_score_pct: float) -> tuple[str, str, str]:
    if fake_score_pct >= 70:
        return "High Risk", "Likely Fake / Misleading", "risk-high"
    if fake_score_pct >= 45:
        return "Moderate Risk", "Unclear / Needs Verification", "risk-mid"
    return "Low Risk", "Likely Credible", "risk-low"


def analyze_news(headline: str, article_text: str, source_url: str) -> dict[str, Any]:
    headline = headline.strip()
    article_text = article_text.strip()
    source_url = source_url.strip()
    combined_text = f"{headline} {article_text}".strip()
    domain = extract_domain(source_url)

    ml_prob = get_ml_fake_probability(combined_text)

    suspicious_hits = count_matches(combined_text, SUSPICIOUS_PHRASES)
    caps_ratio = all_caps_ratio(headline)
    punctuation_score = excessive_punctuation_score(f"{headline} {article_text}")
    evidence_score = evidence_strength(article_text)
    bias_score = emotional_bias_score(combined_text)
    readability = readability_risk(article_text)
    trusted_domain = looks_trusted(domain)
    uses_https = source_url.startswith("https://") if source_url else False

    red_flags: list[str] = []
    positive_signals: list[str] = []

    if suspicious_hits:
        red_flags.append(
            f"Sensational phrase patterns detected: {', '.join(suspicious_hits[:4])}"
        )
    if caps_ratio > ALL_CAPS_THRESHOLD:
        red_flags.append("Headline uses excessive ALL CAPS for emphasis")
    if punctuation_score > 0.4:
        red_flags.append("Heavy dramatic punctuation can indicate clickbait framing")
    if evidence_score < 0.25:
        red_flags.append("Weak sourcing language: claims appear under-cited")
    if bias_score > 0.35:
        red_flags.append("Emotionally or politically loaded wording is elevated")

    if source_url and not trusted_domain:
        red_flags.append("Source domain is not in trusted baseline list")
    if source_url and not uses_https:
        red_flags.append("Source URL does not use HTTPS")

    if evidence_score >= 0.35:
        positive_signals.append("Article includes attribution/evidence-style language")
    if trusted_domain:
        positive_signals.append(f"Source domain has strong institutional trust signals ({domain})")
    if caps_ratio < 0.15 and punctuation_score < 0.20:
        positive_signals.append("Tone appears less sensational (limited caps/punctuation abuse)")

    source_risk = 0.20
    if source_url:
        source_risk = 0.38 if not trusted_domain else 0.06
        if not uses_https:
            source_risk += 0.08
    source_risk = clamp(source_risk)

    style_risk = clamp(0.30 * punctuation_score + 0.35 * caps_ratio + 0.35 * readability)
    claim_risk = clamp(1.0 - evidence_score)

    suspicious_risk = clamp(min(len(suspicious_hits), 6) / 6.0)

    weighted_rule = (
        0.28 * suspicious_risk
        + 0.25 * claim_risk
        + 0.20 * source_risk
        + 0.15 * style_risk
        + 0.12 * bias_score
    )

    final_score = clamp((0.62 * ml_prob) + (0.38 * weighted_rule))
    fake_score_pct = round(final_score * 100, 2)

    risk_band, predicted_label, risk_class = get_risk_band(fake_score_pct)

    # Confidence increases when signals align and distance from center grows.
    signal_alignment = abs(ml_prob - weighted_rule)
    confidence = clamp((abs(final_score - 0.5) * 2.0) + (0.15 * (1 - signal_alignment)))

    breakdown = {
        "ML Pattern Risk": round(ml_prob * 100, 1),
        "Source Credibility Risk": round(source_risk * 100, 1),
        "Evidence Weakness Risk": round(claim_risk * 100, 1),
        "Style/Sensational Risk": round(style_risk * 100, 1),
        "Bias/Emotion Risk": round(bias_score * 100, 1),
    }

    verification_steps = [
        "Cross-check key claim with at least two independent, high-credibility outlets.",
        "Locate primary source documents (official reports, filings, or datasets).",
        "Verify publication date, author identity, and whether headline matches article body.",
    ]

    if source_url and not trusted_domain:
        verification_steps.insert(0, "Treat source as unverified until corroborated by established publishers.")
    if evidence_score < 0.25:
        verification_steps.insert(0, "Look for missing citations: claims without named sources should be considered tentative.")

    explanation = (
        f"Composite score blends ML language patterns ({ml_prob:.2f}) and forensic rule signals ({weighted_rule:.2f}). "
        f"Evidence strength={'high' if evidence_score >= 0.35 else 'low'}, "
        f"source trust={'strong' if trusted_domain else 'unverified' if domain else 'not supplied'}, "
        f"sensationality={'elevated' if suspicious_risk > 0.35 or style_risk > 0.35 else 'controlled'}."
    )

    return {
        "headline": headline,
        "article_text": article_text,
        "source_url": source_url,
        "domain": domain,
        "predicted_label": predicted_label,
        "risk_band": risk_band,
        "risk_class": risk_class,
        "fake_score": fake_score_pct,
        "confidence": round(confidence * 100, 2),
        "red_flags": red_flags if red_flags else ["No major red flags detected by current heuristics."],
        "positive_signals": positive_signals if positive_signals else ["No strong credibility signals detected."],
        "breakdown": breakdown,
        "verification_steps": verification_steps[:4],
        "explanation": explanation,
    }


def save_analysis(result: dict[str, Any]) -> None:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO analysis_history
        (headline, article_text, source_url, source_domain, predicted_label, risk_band,
         fake_score, confidence, explanation, red_flags_json, positive_signals_json,
         breakdown_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            result["headline"],
            result["article_text"],
            result["source_url"],
            result["domain"],
            result["predicted_label"],
            result["risk_band"],
            result["fake_score"],
            result["confidence"],
            result["explanation"],
            json.dumps(result["red_flags"]),
            json.dumps(result["positive_signals"]),
            json.dumps(result["breakdown"]),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    conn.commit()
    conn.close()


def fetch_history(limit: int = 10) -> list[dict[str, Any]]:
    conn = get_db_connection(row_factory=True)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT headline, source_domain, predicted_label, risk_band, fake_score, confidence, created_at
        FROM analysis_history
        ORDER BY id DESC
        LIMIT ?
    """,
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def fetch_dashboard_stats() -> dict[str, Any]:
    conn = get_db_connection(row_factory=True)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) AS total, AVG(fake_score) AS avg_score FROM analysis_history")
    row = cur.fetchone()
    total = row["total"] or 0
    avg_score = round(float(row["avg_score"] or 0), 1)

    cur.execute(
        "SELECT COUNT(*) AS high_risk FROM analysis_history WHERE fake_score >= 70"
    )
    high_risk = cur.fetchone()["high_risk"] or 0

    last_24h = (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
    cur.execute(
        "SELECT COUNT(*) AS recent FROM analysis_history WHERE created_at >= ?",
        (last_24h,),
    )
    recent = cur.fetchone()["recent"] or 0

    conn.close()

    high_risk_rate = round((high_risk / total) * 100, 1) if total else 0.0
    return {
        "total_scans": total,
        "avg_fake_score": avg_score,
        "high_risk_rate": high_risk_rate,
        "recent_scans": recent,
    }


# -----------------------------
# HTML template
# -----------------------------
PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PulseVerify | Decision Intelligence</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&family=IBM+Plex+Mono:wght@500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-main: #f3f2ee;
            --bg-rail: rgba(255, 255, 255, 0.74);
            --bg-panel: rgba(255, 255, 255, 0.78);
            --line: rgba(101, 114, 143, 0.12);
            --line-strong: rgba(92, 111, 164, 0.20);
            --text-main: #1f2a3a;
            --text-muted: #59697f;
            --text-soft: #7d8ca0;
            --safe: #2e8b6c;
            --warn: #cf8e25;
            --risk: #d45c74;
            --accent: #243a73;
            --accent-2: #ff7a59;
            --accent-3: #13b5c8;
            --news-ink: #2f241f;
            --shadow: 0 20px 44px rgba(86, 99, 120, 0.10);
        }

        * { box-sizing: border-box; }

        body {
            margin: 0;
            min-height: 100vh;
            background:
                radial-gradient(900px 520px at 0% 0%, rgba(199, 212, 226, 0.45), transparent 60%),
                radial-gradient(760px 420px at 100% 10%, rgba(255, 209, 195, 0.45), transparent 58%),
                radial-gradient(700px 500px at 70% 100%, rgba(176, 236, 241, 0.36), transparent 60%),
                linear-gradient(180deg, rgba(255,255,255,0.36) 1px, transparent 1px) 0 0 / 100% 48px,
                linear-gradient(90deg, rgba(255,255,255,0.22) 1px, transparent 1px) 0 0 / 48px 100%,
                linear-gradient(180deg, #f7f9fc, var(--bg-main));
            color: var(--text-main);
            font-family: "DM Sans", sans-serif;
        }

        .app-shell {
            width: min(1360px, 96vw);
            margin: 0 auto;
            padding: 24px 0 38px;
            display: grid;
            grid-template-columns: 250px 1fr;
            gap: 28px;
        }

        .rail {
            border: 1px solid rgba(255, 255, 255, 0.45);
            background: linear-gradient(180deg, rgba(255,255,255,0.76), var(--bg-rail));
            border-radius: 22px;
            box-shadow: var(--shadow);
            padding: 22px 18px;
            position: sticky;
            top: 20px;
            height: calc(100vh - 44px);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            backdrop-filter: blur(20px);
        }

        .brand {
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.84rem;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            color: var(--accent);
            margin-bottom: 26px;
        }

        .rail-section {
            margin-bottom: 18px;
        }

        .rail-label {
            font-size: 0.67rem;
            color: var(--text-soft);
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .rail-item {
            display: block;
            width: 100%;
            border: 1px solid transparent;
            border-radius: 14px;
            padding: 10px 12px;
            margin-bottom: 8px;
            background: transparent;
            color: var(--text-muted);
            font-size: 0.82rem;
            text-align: left;
        }

        .rail-item.active {
            border-color: var(--line-strong);
            background: linear-gradient(135deg, rgba(239, 244, 255, 0.95), rgba(255, 241, 236, 0.9));
            color: var(--text-main);
            box-shadow: 0 8px 20px rgba(104, 120, 145, 0.10);
        }

        .rail-foot {
            border-top: 1px solid var(--line);
            padding-top: 16px;
            color: var(--text-soft);
            font-size: 0.74rem;
            line-height: 1.5;
        }

        .workspace {
            display: grid;
            grid-template-rows: auto auto auto 1fr;
            gap: 18px;
        }

        .toolbar {
            border: none;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.50);
            padding: 8px 2px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
        }

        .toolbar-left {
            display: flex;
            gap: 8px;
            align-items: center;
            flex-wrap: wrap;
        }

        .chip {
            border: 1px solid rgba(126, 147, 176, 0.14);
            border-radius: 999px;
            padding: 7px 12px;
            font-size: 0.68rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--text-muted);
            background: rgba(255,255,255,0.72);
        }

        .timestamp {
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.72rem;
            color: var(--text-soft);
        }

        .hero {
            border: none;
            border-radius: 28px;
            background:
                radial-gradient(380px 220px at 100% 0%, rgba(255, 213, 196, 0.75), transparent 70%),
                radial-gradient(320px 220px at 0% 0%, rgba(182, 232, 237, 0.60), transparent 72%),
                linear-gradient(180deg, rgba(255,255,255,0.92), rgba(252,249,245,0.82));
            padding: 28px 30px;
            box-shadow: var(--shadow);
            position: relative;
            overflow: hidden;
        }

        .hero::after {
            content: "";
            position: absolute;
            left: 30px;
            right: 30px;
            top: 64px;
            height: 1px;
            background: linear-gradient(90deg, rgba(36,58,115,0.22), rgba(255,122,89,0.12), transparent);
        }

        .hero h1 {
            margin: 10px 0 12px;
            font-size: clamp(1.75rem, 2.9vw, 2.6rem);
            line-height: 1.1;
            letter-spacing: -0.014em;
            font-weight: 700;
            max-width: 880px;
            font-family: "Fraunces", serif;
            color: var(--news-ink);
        }

        .hero p {
            margin: 0;
            color: var(--text-muted);
            max-width: 1000px;
            font-size: 1rem;
        }

        .hero-kicker {
            font-family: "IBM Plex Mono", monospace;
            color: var(--accent);
            font-size: 0.7rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
        }

        .hero-meta {
            margin-top: 18px;
            display: flex;
            gap: 14px;
            flex-wrap: wrap;
            color: var(--text-soft);
            font-size: 0.82rem;
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
        }

        .stat {
            border: none;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.82), rgba(255, 255, 255, 0.58));
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: 0 10px 26px rgba(98, 112, 133, 0.08);
        }

        .stat-label {
            font-size: 0.66rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--text-soft);
        }

        .stat-value {
            margin-top: 7px;
            font-family: "IBM Plex Mono", monospace;
            font-size: 1.04rem;
            color: var(--accent);
        }

        .modules {
            display: grid;
            grid-template-columns: 1.1fr 0.9fr;
            gap: 18px;
        }

        .stack {
            display: grid;
            grid-template-rows: auto auto;
            gap: 18px;
        }

        .panel {
            border: none;
            background: var(--bg-panel);
            border-radius: 24px;
            box-shadow: var(--shadow);
            padding: 24px;
            backdrop-filter: blur(14px);
        }

        .analyzer-panel {
            background:
                radial-gradient(260px 180px at 100% 0%, rgba(19, 181, 200, 0.12), transparent 70%),
                linear-gradient(180deg, rgba(255,255,255,0.82), rgba(252,252,251,0.75));
        }

        .chat-panel {
            background:
                linear-gradient(180deg, rgba(245, 248, 255, 0.92), rgba(255, 252, 247, 0.88));
        }

        .panel-head {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 10px;
            margin-bottom: 12px;
        }

        .panel h2 {
            margin: 0;
            font-size: 1rem;
            letter-spacing: 0.01em;
            color: var(--accent);
        }

        .panel-sub {
            margin: 0;
            color: var(--text-soft);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }

        label {
            display: block;
            margin: 12px 0 6px;
            color: #617389;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-size: 0.69rem;
        }

        input[type="text"],
        textarea {
            width: 100%;
            border-radius: 16px;
            border: 1px solid rgba(167, 188, 219, 0.22);
            background: rgba(248, 250, 253, 0.92);
            color: #0d1628;
            font-size: 0.9rem;
            padding: 14px 15px;
            font-family: "DM Sans", sans-serif;
            outline: none;
        }

        textarea {
            min-height: 176px;
            resize: vertical;
        }

        input:focus, textarea:focus {
            box-shadow: 0 0 0 3px rgba(202, 215, 232, 0.72);
        }

        .btn-row {
            margin-top: 12px;
            display: flex;
            gap: 9px;
            flex-wrap: wrap;
        }

        .btn {
            border-radius: 16px;
            border: 1px solid transparent;
            padding: 12px 16px;
            font-family: "DM Sans", sans-serif;
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            cursor: pointer;
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--accent), #4257a4 60%, var(--accent-3));
            color: #f7faff;
            flex: 1;
            min-width: 230px;
            box-shadow: 0 14px 28px rgba(60, 88, 162, 0.22);
        }

        .btn-secondary {
            background: rgba(255,255,255,0.82);
            border-color: rgba(126, 147, 176, 0.12);
            color: var(--accent);
        }

        .disclaimer {
            margin-top: 11px;
            color: var(--text-muted);
            font-size: 0.76rem;
        }

        .risk-pill {
            display: inline-flex;
            border-radius: 999px;
            padding: 5px 11px;
            font-size: 0.68rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 700;
            margin-bottom: 10px;
            border: 1px solid;
        }

        .risk-high { color: #9d4f4c; border-color: rgba(201,109,103,0.24); background: rgba(201,109,103,0.10); }
        .risk-mid { color: #8f6628; border-color: rgba(183,130,50,0.24); background: rgba(183,130,50,0.10); }
        .risk-low { color: #49735c; border-color: rgba(93,138,113,0.24); background: rgba(93,138,113,0.10); }

        .chat-thread {
            display: grid;
            gap: 12px;
        }

        .bubble {
            padding: 14px 16px;
            border-radius: 20px;
            line-height: 1.5;
            font-size: 0.9rem;
            max-width: 100%;
        }

        .bubble-assistant {
            background: linear-gradient(180deg, rgba(234, 241, 255, 0.92), rgba(255, 248, 243, 0.88));
            color: var(--text-main);
            border-top-left-radius: 8px;
        }

        .bubble-system {
            background: rgba(255,255,255,0.72);
            color: var(--text-muted);
            border: 1px solid rgba(126, 147, 176, 0.10);
            border-top-right-radius: 8px;
        }

        .bubble-title {
            display: block;
            margin-bottom: 6px;
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.66rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--accent);
        }

        .score-wrap {
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 12px;
            align-items: center;
            margin-bottom: 12px;
        }

        .score-ring {
            --score: {{ result.fake_score if result else 0 }};
            width: 116px;
            height: 116px;
            border-radius: 50%;
            background: conic-gradient(var(--accent) calc(var(--score) * 1%), rgba(160, 175, 194, 0.16) 0);
            display: grid;
            place-items: center;
        }

        .score-inner {
            width: 84px;
            height: 84px;
            border-radius: 50%;
            background: rgba(249, 251, 254, 0.96);
            display: grid;
            place-items: center;
            text-align: center;
            box-shadow: inset 0 0 0 1px rgba(166, 183, 206, 0.16);
        }

        .score-val {
            font-family: "IBM Plex Mono", monospace;
            font-size: 1rem;
            color: var(--accent);
        }

        .score-label {
            font-size: 0.65rem;
            color: var(--text-soft);
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .meta {
            color: var(--text-muted);
            font-size: 0.84rem;
            line-height: 1.45;
        }

        .section-title {
            margin: 14px 0 7px;
            font-size: 0.68rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #60758d;
        }

        .bar-row { margin-bottom: 8px; }

        .bar-head {
            display: flex;
            justify-content: space-between;
            font-size: 0.79rem;
            margin-bottom: 4px;
            color: var(--text-main);
        }

        .bar-track {
            height: 8px;
            background: rgba(180, 194, 214, 0.18);
            border-radius: 999px;
            overflow: hidden;
        }

        .bar-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--accent), #6d82d0, var(--accent-2));
        }

        .list {
            margin: 0;
            padding-left: 17px;
            color: var(--text-main);
            font-size: 0.84rem;
        }

        .list li { margin-bottom: 6px; }

        .protocol-note {
            color: var(--text-muted);
            font-size: 0.82rem;
            line-height: 1.45;
        }

        .history-row {
            border: 1px solid rgba(130, 148, 171, 0.10);
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.78), rgba(252, 248, 244, 0.60));
            padding: 14px 15px;
            margin-bottom: 8px;
        }

        .history-top {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .history-meta {
            margin-top: 5px;
            color: var(--text-soft);
            font-size: 0.75rem;
        }

        .empty {
            color: var(--text-muted);
            font-size: 0.84rem;
        }

        .form-status {
            min-height: 20px;
            margin-top: 10px;
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        .form-status.error {
            color: var(--risk);
        }

        @media (max-width: 1160px) {
            .app-shell {
                grid-template-columns: 1fr;
            }

            .rail {
                position: static;
                height: auto;
            }

            .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .modules { grid-template-columns: 1fr; }
        }

        @media (max-width: 700px) {
            .app-shell {
                width: min(96vw, 96vw);
                padding: 10px 0 20px;
            }

            .stats { grid-template-columns: 1fr; }
            .score-wrap { grid-template-columns: 1fr; text-align: center; justify-items: center; }
            .toolbar { flex-direction: column; align-items: flex-start; }
            .hero::after { left: 18px; right: 18px; top: 58px; }
        }
    </style>
</head>
<body>
    <main class="app-shell">
        <aside class="rail">
            <div>
                <div class="brand">PulseVerify</div>
                <div class="rail-section">
                    <div class="rail-label">Modules</div>
                    <button class="rail-item active" type="button">Decision Dashboard</button>
                    <button class="rail-item" type="button">Story Analyzer</button>
                    <button class="rail-item" type="button">Risk Breakdown</button>
                    <button class="rail-item" type="button">Verification Protocol</button>
                    <button class="rail-item" type="button">Scan History</button>
                </div>
                <div class="rail-section">
                    <div class="rail-label">System</div>
                    <div class="rail-item">Model: Hybrid Heuristics + NLP</div>
                    <div class="rail-item">Region: United States</div>
                    <div class="rail-item">Mode: Analyst Workspace</div>
                </div>
            </div>
            <div class="rail-foot">
                Educational deployment.<br>
                Final truth judgments require independent reporting and primary evidence review.
            </div>
        </aside>

        <section class="workspace">
            <section class="toolbar">
                <div class="toolbar-left">
                    <span class="chip">Decision Intelligence</span>
                    <span class="chip">Misinformation Detection</span>
                </div>
                <div class="timestamp" id="timestamp">{{ stats.recent_scans }} scans in the last 24 hours</div>
            </section>

            <section class="hero">
                <div class="hero-kicker">Morning Edition / Fact Check Desk</div>
                <h1>Detect misinformation with auditable scoring, source scrutiny, and evidence diagnostics.</h1>
                <p>
                    Evaluate headlines and body text through a structured model that combines machine-learning pattern detection,
                    source credibility heuristics, and operator-ready verification guidance.
                </p>
                <div class="hero-meta">
                    <span>Editorial AI workspace</span>
                    <span>Conversational analysis</span>
                    <span>Source-aware reporting</span>
                </div>
            </section>

            <section class="stats">
                <article class="stat">
                    <div class="stat-label">Total Scans</div>
                    <div class="stat-value" id="stat-total">{{ stats.total_scans }}</div>
                </article>
                <article class="stat">
                    <div class="stat-label">Average Risk</div>
                    <div class="stat-value" id="stat-avg">{{ stats.avg_fake_score }}%</div>
                </article>
                <article class="stat">
                    <div class="stat-label">High-Risk Share</div>
                    <div class="stat-value" id="stat-high">{{ stats.high_risk_rate }}%</div>
                </article>
                <article class="stat">
                    <div class="stat-label">Scans (24h)</div>
                    <div class="stat-value" id="stat-recent">{{ stats.recent_scans }}</div>
                </article>
            </section>

            <section class="modules">
                <article class="panel analyzer-panel">
                    <div class="panel-head">
                        <h2>Story Analyzer</h2>
                        <p class="panel-sub">Newsroom Input</p>
                    </div>

                    <form method="POST" action="/" id="analyzer-form">
                        <label for="headline">Headline</label>
                        <input type="text" id="headline" name="headline" required
                            placeholder="Paste headline"
                            value="{{ request.form.get('headline', '') }}">

                        <label for="article_text">Article Text</label>
                        <textarea id="article_text" name="article_text" required
                            placeholder="Paste full article text or a detailed excerpt">{{ request.form.get('article_text', '') }}</textarea>

                        <label for="source_url">Source URL (optional)</label>
                        <input type="text" id="source_url" name="source_url"
                            placeholder="https://example.com/news/story"
                            value="{{ request.form.get('source_url', '') }}">

                        <div class="btn-row">
                            <button class="btn btn-primary" id="scan-button" type="submit">Run Fake News Scan</button>
                            <button class="btn btn-secondary" id="demo-button" type="button" onclick="fillDemo()">Use Demo Example</button>
                        </div>
                    </form>
                    <div class="form-status" id="form-status"></div>

                    <div class="disclaimer">
                        Educational detector. Always verify critical claims with primary sources and trusted publishers.
                    </div>
                    {% if error_message %}
                        <div class="form-status error">{{ error_message }}</div>
                    {% endif %}
                </article>

                <div class="stack">
                    <article class="panel chat-panel">
                        <div class="panel-head">
                            <h2>Assistant Briefing</h2>
                            <p class="panel-sub">Analysis Thread</p>
                        </div>

                        <div id="result-panel-body">
                        {% if result %}
                            <div class="chat-thread">
                                <div class="bubble bubble-assistant">
                                    <span class="bubble-title">Assistant Summary</span>
                                    <span class="risk-pill {{ result.risk_class }}">{{ result.risk_band }}</span>
                                    <div class="score-wrap">
                                        <div class="score-ring">
                                            <div class="score-inner">
                                                <div class="score-val">{{ result.fake_score }}%</div>
                                                <div class="score-label">fake risk</div>
                                            </div>
                                        </div>
                                        <div class="meta">
                                            <strong>{{ result.predicted_label }}</strong><br>
                                            Confidence: <strong>{{ result.confidence }}%</strong><br>
                                            Source: <strong>{{ result.domain if result.domain else 'Not provided' }}</strong><br>
                                            {{ result.explanation }}
                                        </div>
                                    </div>
                                </div>

                                <div class="bubble bubble-system">
                                    <span class="bubble-title">Why It Was Flagged</span>
                                    <div class="section-title">Risk Breakdown</div>
                                    {% for metric, val in result.breakdown.items() %}
                                        <div class="bar-row">
                                            <div class="bar-head"><span>{{ metric }}</span><span>{{ val }}%</span></div>
                                            <div class="bar-track"><div class="bar-fill" style="width: {{ val }}%"></div></div>
                                        </div>
                                    {% endfor %}
                                </div>

                                <div class="bubble bubble-assistant">
                                    <span class="bubble-title">Editorial Signals</span>
                                    <div class="section-title">Red Flags</div>
                                    <ul class="list">
                                        {% for flag in result.red_flags %}
                                            <li>{{ flag }}</li>
                                        {% endfor %}
                                    </ul>

                                    <div class="section-title">Positive Signals</div>
                                    <ul class="list">
                                        {% for signal in result.positive_signals %}
                                            <li>{{ signal }}</li>
                                        {% endfor %}
                                    </ul>
                                </div>
                            </div>
                        {% else %}
                            <p class="empty">No scan yet. Submit a story to generate a full fake-news risk report.</p>
                        {% endif %}
                        </div>
                    </article>

                    <article class="panel chat-panel">
                        <div class="panel-head">
                            <h2>Reporter Notes</h2>
                            <p class="panel-sub">Verification Checklist</p>
                        </div>

                        <div id="verification-panel-body">
                        {% if result %}
                            <ul class="list">
                                {% for step in result.verification_steps %}
                                    <li>{{ step }}</li>
                                {% endfor %}
                            </ul>
                        {% else %}
                            <p class="protocol-note">
                                Run an analysis to generate a tailored verification sequence. Protocol emphasizes source triangulation,
                                primary-document validation, and claim-timeline checks.
                            </p>
                        {% endif %}
                        </div>
                    </article>
                </div>
            </section>

            <section class="panel">
                <div class="panel-head">
                    <h2>Recent Scan History</h2>
                    <p class="panel-sub">Local Session Log</p>
                </div>
                <div id="history-list">
                {% if history %}
                    {% for item in history %}
                        <div class="history-row">
                            <div class="history-top">
                                <span>{{ item.headline }}</span>
                                <span>{{ item.fake_score }}%</span>
                            </div>
                            <div class="history-meta">
                                {{ item.risk_band }} | {{ item.predicted_label }} | Confidence {{ item.confidence }}% |
                                {{ item.source_domain if item.source_domain else 'source n/a' }} | {{ item.created_at }}
                            </div>
                        </div>
                    {% endfor %}
                {% else %}
                    <p class="empty">No history yet.</p>
                {% endif %}
                </div>
            </section>
        </section>
    </main>

    <script>
        const STORAGE_KEY = "pulseverify-history-v1";
        const TRUSTED_DOMAINS = new Set([
            "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "npr.org", "nytimes.com",
            "wsj.com", "theguardian.com", "cbsnews.com", "abcnews.go.com", "cnn.com",
            "who.int", "cdc.gov", "nih.gov", "un.org"
        ]);
        const SUSPICIOUS_PHRASES = [
            "you will not believe", "they do not want you to know", "they don't want you to know",
            "one weird trick", "share before it gets deleted", "miracle cure", "hidden truth",
            "mainstream media won't tell you", "wake up", "guaranteed proof",
            "secret elites", "anonymous insiders", "100% proof", "shocking", "unbelievable"
        ];
        const PARTISAN_LOADED_WORDS = [
            "traitor", "corrupt", "evil", "radical", "enemy",
            "propaganda", "betrayal", "rigged", "brainwashed", "destroying"
        ];
        const EVIDENCE_WORDS = [
            "according to", "reported by", "study", "data", "source", "official",
            "document", "research", "agency", "court", "peer reviewed", "analysis", "statement"
        ];

        function clamp(value, low = 0, high = 1) {
            return Math.max(low, Math.min(value, high));
        }

        function escapeHtml(value) {
            return String(value)
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#39;");
        }

        function setFormStatus(message, isError = false) {
            const status = document.getElementById("form-status");
            status.textContent = message;
            status.classList.toggle("error", isError);
        }

        function setLoading(isLoading) {
            document.getElementById("scan-button").disabled = isLoading;
            document.getElementById("demo-button").disabled = isLoading;
        }

        function extractDomain(url) {
            if (!url) return "";
            try {
                const normalized = url.startsWith("http://") || url.startsWith("https://") ? url : `https://${url}`;
                return new URL(normalized).hostname.toLowerCase().replace(/^www\\./, "");
            } catch {
                return "";
            }
        }

        function looksTrusted(domain) {
            return Boolean(domain) && (domain.endsWith(".gov") || domain.endsWith(".edu") || TRUSTED_DOMAINS.has(domain));
        }

        function countMatches(text, phrases) {
            const lowered = text.toLowerCase();
            return phrases.filter((phrase) => lowered.includes(phrase));
        }

        function allCapsRatio(text) {
            const letters = Array.from(text).filter((ch) => /[A-Za-z]/.test(ch));
            if (!letters.length) return 0;
            const caps = letters.filter((ch) => ch === ch.toUpperCase());
            return caps.length / letters.length;
        }

        function excessivePunctuationScore(text) {
            const drama = (text.match(/[!?]/g) || []).length;
            return clamp(drama / 10);
        }

        function termStrength(text, terms, divisor) {
            const lowered = text.toLowerCase();
            const hits = terms.reduce((sum, term) => sum + (lowered.split(term).length - 1), 0);
            return clamp(hits / divisor);
        }

        function readabilityRisk(text) {
            const sentences = text.split(/[.!?]+/).map((s) => s.trim()).filter(Boolean);
            const words = text.match(/\\b\\w+\\b/g) || [];
            if (!sentences.length || !words.length) return 0.5;
            const avgWords = words.length / sentences.length;
            if (avgWords < 7) return 0.45;
            if (avgWords > 35) return 0.40;
            return 0.12;
        }

        function heuristicMlProbability(text) {
            const lowered = text.toLowerCase();
            const fakeHits = [...SUSPICIOUS_PHRASES, ...PARTISAN_LOADED_WORDS]
                .reduce((sum, term) => sum + (lowered.split(term).length - 1), 0);
            const realHits = EVIDENCE_WORDS
                .reduce((sum, term) => sum + (lowered.split(term).length - 1), 0);
            const lengthBonus = lowered.split(/\\s+/).filter(Boolean).length < 45 ? 0.08 : 0.0;
            const score = 0.48 + Math.min(fakeHits * 0.06, 0.32) - Math.min(realHits * 0.05, 0.28) + lengthBonus;
            return clamp(score, 0.05, 0.95);
        }

        function getRiskBand(fakeScorePct) {
            if (fakeScorePct >= 70) return ["High Risk", "Likely Fake / Misleading", "risk-high"];
            if (fakeScorePct >= 45) return ["Moderate Risk", "Unclear / Needs Verification", "risk-mid"];
            return ["Low Risk", "Likely Credible", "risk-low"];
        }

        function analyzeNewsClient(headline, articleText, sourceUrl) {
            const combinedText = `${headline} ${articleText}`.trim();
            const domain = extractDomain(sourceUrl);
            const mlProb = heuristicMlProbability(combinedText);
            const suspiciousHits = countMatches(combinedText, SUSPICIOUS_PHRASES);
            const capsRatio = allCapsRatio(headline);
            const punctuationScore = excessivePunctuationScore(`${headline} ${articleText}`);
            const evidenceScore = termStrength(articleText, EVIDENCE_WORDS, 8);
            const biasScore = termStrength(combinedText, PARTISAN_LOADED_WORDS, 8);
            const readability = readabilityRisk(articleText);
            const trustedDomain = looksTrusted(domain);
            const usesHttps = sourceUrl ? sourceUrl.startsWith("https://") : false;

            const redFlags = [];
            const positiveSignals = [];

            if (suspiciousHits.length) redFlags.push(`Sensational phrase patterns detected: ${suspiciousHits.slice(0, 4).join(", ")}`);
            if (capsRatio > 0.33) redFlags.push("Headline uses excessive ALL CAPS for emphasis");
            if (punctuationScore > 0.4) redFlags.push("Heavy dramatic punctuation can indicate clickbait framing");
            if (evidenceScore < 0.25) redFlags.push("Weak sourcing language: claims appear under-cited");
            if (biasScore > 0.35) redFlags.push("Emotionally or politically loaded wording is elevated");
            if (sourceUrl && !trustedDomain) redFlags.push("Source domain is not in trusted baseline list");
            if (sourceUrl && !usesHttps) redFlags.push("Source URL does not use HTTPS");

            if (evidenceScore >= 0.35) positiveSignals.push("Article includes attribution/evidence-style language");
            if (trustedDomain) positiveSignals.push(`Source domain has strong institutional trust signals (${domain})`);
            if (capsRatio < 0.15 && punctuationScore < 0.20) positiveSignals.push("Tone appears less sensational (limited caps/punctuation abuse)");

            let sourceRisk = 0.20;
            if (sourceUrl) {
                sourceRisk = trustedDomain ? 0.06 : 0.38;
                if (!usesHttps) sourceRisk += 0.08;
            }
            sourceRisk = clamp(sourceRisk);

            const styleRisk = clamp(0.30 * punctuationScore + 0.35 * capsRatio + 0.35 * readability);
            const claimRisk = clamp(1 - evidenceScore);
            const suspiciousRisk = clamp(Math.min(suspiciousHits.length, 6) / 6);
            const weightedRule = (
                0.28 * suspiciousRisk +
                0.25 * claimRisk +
                0.20 * sourceRisk +
                0.15 * styleRisk +
                0.12 * biasScore
            );
            const finalScore = clamp((0.62 * mlProb) + (0.38 * weightedRule));
            const fakeScore = Math.round(finalScore * 10000) / 100;
            const [riskBand, predictedLabel, riskClass] = getRiskBand(fakeScore);
            const signalAlignment = Math.abs(mlProb - weightedRule);
            const confidence = Math.round(clamp((Math.abs(finalScore - 0.5) * 2.0) + (0.15 * (1 - signalAlignment))) * 10000) / 100;

            const verificationSteps = [
                "Cross-check key claim with at least two independent, high-credibility outlets.",
                "Locate primary source documents (official reports, filings, or datasets).",
                "Verify publication date, author identity, and whether headline matches article body."
            ];
            if (sourceUrl && !trustedDomain) verificationSteps.unshift("Treat source as unverified until corroborated by established publishers.");
            if (evidenceScore < 0.25) verificationSteps.unshift("Look for missing citations: claims without named sources should be considered tentative.");

            return {
                headline,
                article_text: articleText,
                source_url: sourceUrl,
                domain,
                predicted_label: predictedLabel,
                risk_band: riskBand,
                risk_class: riskClass,
                fake_score: fakeScore,
                confidence,
                red_flags: redFlags.length ? redFlags : ["No major red flags detected by current heuristics."],
                positive_signals: positiveSignals.length ? positiveSignals : ["No strong credibility signals detected."],
                breakdown: {
                    "ML Pattern Risk": Math.round(mlProb * 1000) / 10,
                    "Source Credibility Risk": Math.round(sourceRisk * 1000) / 10,
                    "Evidence Weakness Risk": Math.round(claimRisk * 1000) / 10,
                    "Style/Sensational Risk": Math.round(styleRisk * 1000) / 10,
                    "Bias/Emotion Risk": Math.round(biasScore * 1000) / 10
                },
                verification_steps: verificationSteps.slice(0, 4),
                explanation: `Composite score blends heuristic language patterns (${mlProb.toFixed(2)}) and forensic rule signals (${weightedRule.toFixed(2)}). Evidence strength=${evidenceScore >= 0.35 ? "high" : "low"}, source trust=${trustedDomain ? "strong" : domain ? "unverified" : "not supplied"}, sensationality=${(suspiciousRisk > 0.35 || styleRisk > 0.35) ? "elevated" : "controlled"}.`
            };
        }

        function loadStoredHistory() {
            try {
                return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
            } catch {
                return [];
            }
        }

        function saveStoredHistory(history) {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(history.slice(0, 10)));
        }

        function buildStats(history) {
            const total = history.length;
            const avg = total ? (history.reduce((sum, item) => sum + Number(item.fake_score || 0), 0) / total) : 0;
            const high = total ? (history.filter((item) => Number(item.fake_score) >= 70).length / total) * 100 : 0;
            return {
                total_scans: total,
                avg_fake_score: Math.round(avg * 10) / 10,
                high_risk_rate: Math.round(high * 10) / 10,
                recent_scans: total
            };
        }

        function renderResult(result) {
            const resultPanel = document.getElementById("result-panel-body");
            const verificationPanel = document.getElementById("verification-panel-body");

            const breakdownHtml = Object.entries(result.breakdown).map(([metric, val]) => `
                <div class="bar-row">
                    <div class="bar-head"><span>${escapeHtml(metric)}</span><span>${escapeHtml(val)}%</span></div>
                    <div class="bar-track"><div class="bar-fill" style="width: ${Number(val)}%"></div></div>
                </div>
            `).join("");

            const redFlagsHtml = result.red_flags.map((flag) => `<li>${escapeHtml(flag)}</li>`).join("");
            const positiveSignalsHtml = result.positive_signals.map((signal) => `<li>${escapeHtml(signal)}</li>`).join("");
            const verificationHtml = result.verification_steps.map((step) => `<li>${escapeHtml(step)}</li>`).join("");

            resultPanel.innerHTML = `
                <div class="chat-thread">
                    <div class="bubble bubble-assistant">
                        <span class="bubble-title">Assistant Summary</span>
                        <span class="risk-pill ${escapeHtml(result.risk_class)}">${escapeHtml(result.risk_band)}</span>
                        <div class="score-wrap">
                            <div class="score-ring" style="--score: ${Number(result.fake_score)};">
                                <div class="score-inner">
                                    <div class="score-val">${escapeHtml(result.fake_score)}%</div>
                                    <div class="score-label">fake risk</div>
                                </div>
                            </div>
                            <div class="meta">
                                <strong>${escapeHtml(result.predicted_label)}</strong><br>
                                Confidence: <strong>${escapeHtml(result.confidence)}%</strong><br>
                                Source: <strong>${escapeHtml(result.domain || "Not provided")}</strong><br>
                                ${escapeHtml(result.explanation)}
                            </div>
                        </div>
                    </div>

                    <div class="bubble bubble-system">
                        <span class="bubble-title">Why It Was Flagged</span>
                        <div class="section-title">Risk Breakdown</div>
                        ${breakdownHtml}
                    </div>

                    <div class="bubble bubble-assistant">
                        <span class="bubble-title">Editorial Signals</span>
                        <div class="section-title">Red Flags</div>
                        <ul class="list">${redFlagsHtml}</ul>
                        <div class="section-title">Positive Signals</div>
                        <ul class="list">${positiveSignalsHtml}</ul>
                    </div>
                </div>
            `;

            verificationPanel.innerHTML = `<ul class="list">${verificationHtml}</ul>`;
        }

        function renderHistory(history) {
            const historyList = document.getElementById("history-list");
            if (!history.length) {
                historyList.innerHTML = '<p class="empty">No history yet.</p>';
                return;
            }

            historyList.innerHTML = history.map((item) => `
                <div class="history-row">
                    <div class="history-top">
                        <span>${escapeHtml(item.headline)}</span>
                        <span>${escapeHtml(item.fake_score)}%</span>
                    </div>
                    <div class="history-meta">
                        ${escapeHtml(item.risk_band)} | ${escapeHtml(item.predicted_label)} | Confidence ${escapeHtml(item.confidence)}% |
                        ${escapeHtml(item.source_domain || "source n/a")} | ${escapeHtml(item.created_at)}
                    </div>
                </div>
            `).join("");
        }

        function renderStats(stats) {
            document.getElementById("stat-total").textContent = stats.total_scans;
            document.getElementById("stat-avg").textContent = `${stats.avg_fake_score}%`;
            document.getElementById("stat-high").textContent = `${stats.high_risk_rate}%`;
            document.getElementById("stat-recent").textContent = stats.recent_scans;
            document.getElementById("timestamp").textContent = `${stats.recent_scans} scans in the last 24 hours`;
        }

        function refreshHistoryAndStats() {
            const history = loadStoredHistory();
            renderHistory(history);
            renderStats(buildStats(history));
        }

        function submitAnalysis() {
            const headline = document.getElementById("headline").value.trim();
            const articleText = document.getElementById("article_text").value.trim();
            const sourceUrl = document.getElementById("source_url").value.trim();

            if (!headline || !articleText) {
                setFormStatus("Headline and article text are required.", true);
                return;
            }

            setLoading(true);
            setFormStatus("Analyzing story...");

            try {
                const payload = analyzeNewsClient(headline, articleText, sourceUrl);
                renderResult(payload);

                const history = loadStoredHistory();
                history.unshift({
                    headline: payload.headline,
                    source_domain: payload.domain,
                    predicted_label: payload.predicted_label,
                    risk_band: payload.risk_band,
                    fake_score: payload.fake_score,
                    confidence: payload.confidence,
                    created_at: new Date().toLocaleString()
                });
                saveStoredHistory(history);
                refreshHistoryAndStats();
                setFormStatus("Analysis complete.");
            } catch (error) {
                setFormStatus(error.message || "Analysis failed.", true);
            } finally {
                setLoading(false);
            }
        }

        function fillDemo() {
            const headline = "BREAKING: Secret cure hidden by mainstream media shocks everyone!!!";
            const article = "A viral post claims a miracle cure was discovered overnight. " +
                "It cites anonymous insiders but provides no official data or published research. " +
                "The post urges readers to share before deletion and says doctors are hiding the truth.";
            const url = "http://viral-truth-now.example/story";

            document.getElementById("headline").value = headline;
            document.getElementById("article_text").value = article;
            document.getElementById("source_url").value = url;
            submitAnalysis();
        }

        document.getElementById("analyzer-form").addEventListener("submit", function(event) {
            event.preventDefault();
            submitAnalysis();
        });

        refreshHistoryAndStats();
    </script>
</body>
</html>
"""


# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    ensure_app_ready()
    result = None
    error_message = None

    try:
        if request.method == "POST":
            headline = request.form.get("headline", "").strip()
            article_text = request.form.get("article_text", "").strip()
            source_url = request.form.get("source_url", "").strip()

            if headline and article_text:
                result = analyze_news(headline, article_text, source_url)
                save_analysis(result)

        history = fetch_history()
        stats = fetch_dashboard_stats()
    except Exception as exc:
        history = []
        stats = {
            "total_scans": 0,
            "avg_fake_score": 0,
            "high_risk_rate": 0,
            "recent_scans": 0,
        }
        error_message = str(exc)

    return render_template_string(
        PAGE_TEMPLATE,
        result=result,
        history=history,
        stats=stats,
        error_message=error_message,
        request=request,
    )


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    ensure_app_ready()
    try:
        data = request.get_json(force=True, silent=False)
        headline = data.get("headline", "").strip()
        article_text = data.get("article_text", "").strip()
        source_url = data.get("source_url", "").strip()

        if not headline or not article_text:
            return jsonify({"error": "headline and article_text are required"}), 400

        result = analyze_news(headline, article_text, source_url)
        save_analysis(result)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Analysis failed: {exc}"}), 500


@app.route("/api/history", methods=["GET"])
def api_history():
    ensure_app_ready()
    try:
        return jsonify({"history": fetch_history(), "stats": fetch_dashboard_stats()})
    except Exception as exc:
        return jsonify({"error": f"History failed: {exc}"}), 500


# -----------------------------
# App entry
# -----------------------------
if __name__ == "__main__":
    ensure_app_ready()
    app.run(debug=True)
