import streamlit as st
import boto3
import json
import uuid
from datetime import datetime
from botocore.exceptions import ClientError, NoCredentialsError

# ─────────────────────────────────────────────
#  Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PulseVerify | Forensic News Analyst",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&family=IBM+Plex+Mono:wght@500;600&display=swap');

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
.block-container { padding: 1rem 2rem !important; max-width: 1200px !important; }

.stApp {
    background:
        radial-gradient(900px 520px at 0% 0%, rgba(199,212,226,0.45), transparent 60%),
        radial-gradient(760px 420px at 100% 10%, rgba(255,209,195,0.45), transparent 58%),
        radial-gradient(700px 500px at 70% 100%, rgba(176,236,241,0.36), transparent 60%),
        linear-gradient(180deg, #f7f9fc, #f3f2ee) !important;
    font-family: 'DM Sans', sans-serif !important;
}

.pv-hero {
    border-radius: 28px;
    background: radial-gradient(380px 220px at 100% 0%, rgba(255,213,196,0.75), transparent 70%),
        radial-gradient(320px 220px at 0% 0%, rgba(182,232,237,0.60), transparent 72%),
        linear-gradient(180deg, rgba(255,255,255,0.92), rgba(252,249,245,0.82));
    padding: 32px 36px 28px;
    box-shadow: 0 20px 44px rgba(86,99,120,0.10);
    margin-bottom: 22px;
    border: 1px solid rgba(255,255,255,0.6);
}
.pv-brand { font-family:'IBM Plex Mono',monospace; font-size:0.78rem; letter-spacing:0.22em; text-transform:uppercase; color:#243a73; margin-bottom:8px; }
.pv-hero h1 { margin:8px 0 10px; font-size:2.4rem; line-height:1.1; font-weight:700; font-family:'Fraunces',serif; color:#2f241f; }
.pv-hero p  { color:#59697f; font-size:1rem; margin:0; }

.pv-stats { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:22px; }
.pv-stat  { background:rgba(255,255,255,0.82); border:1px solid rgba(255,255,255,0.55); border-radius:18px; padding:18px 20px; box-shadow:0 20px 44px rgba(86,99,120,0.10); }
.pv-stat-val { font-family:'IBM Plex Mono',monospace; font-size:1.8rem; font-weight:600; color:#243a73; line-height:1; }
.pv-stat-lbl { font-size:0.72rem; color:#7d8ca0; text-transform:uppercase; letter-spacing:0.1em; margin-top:5px; }

.pv-input-card { background:rgba(255,255,255,0.82); border:1px solid rgba(255,255,255,0.55); border-radius:22px; padding:28px 32px; box-shadow:0 20px 44px rgba(86,99,120,0.10); margin-bottom:22px; }
.pv-input-label { font-size:0.72rem; color:#7d8ca0; text-transform:uppercase; letter-spacing:0.14em; margin-bottom:8px; font-family:'IBM Plex Mono',monospace; }

.pv-result { background:rgba(255,255,255,0.82); border:1px solid rgba(255,255,255,0.55); border-radius:22px; padding:28px 32px; box-shadow:0 20px 44px rgba(86,99,120,0.10); margin-bottom:22px; }
.pv-score-wrap { display:flex; align-items:center; gap:28px; margin-bottom:24px; flex-wrap:wrap; }
.pv-ring { width:110px; height:110px; border-radius:50%; display:flex; flex-direction:column; align-items:center; justify-content:center; flex-shrink:0; }
.pv-ring-safe { background:linear-gradient(135deg,#d4f4e7,#a8e6cf); border:3px solid #2e8b6c; }
.pv-ring-warn { background:linear-gradient(135deg,#fef3cd,#fde68a); border:3px solid #cf8e25; }
.pv-ring-risk { background:linear-gradient(135deg,#fde8ed,#fbb6c3); border:3px solid #d45c74; }
.pv-ring-val  { font-family:'IBM Plex Mono',monospace; font-size:1.6rem; font-weight:700; line-height:1; }
.pv-ring-lbl  { font-size:0.62rem; text-transform:uppercase; letter-spacing:0.1em; opacity:0.7; margin-top:3px; }
.pv-verdict   { font-family:'Fraunces',serif; font-size:1.5rem; font-weight:700; color:#2f241f; }
.pv-pill { display:inline-block; padding:4px 14px; border-radius:999px; font-size:0.75rem; font-weight:600; margin-top:6px; }
.pv-pill-safe { background:#d4f4e7; color:#2e8b6c; }
.pv-pill-warn { background:#fef3cd; color:#cf8e25; }
.pv-pill-risk { background:#fde8ed; color:#d45c74; }

.pv-section-title { font-size:0.68rem; text-transform:uppercase; letter-spacing:0.14em; color:#7d8ca0; font-family:'IBM Plex Mono',monospace; margin:18px 0 10px; }
.pv-bar-row { margin-bottom:10px; }
.pv-bar-head { display:flex; justify-content:space-between; font-size:0.82rem; color:#59697f; margin-bottom:4px; }
.pv-bar-track { height:7px; background:rgba(101,114,143,0.12); border-radius:999px; overflow:hidden; }
.pv-bar-fill  { height:100%; border-radius:999px; background:linear-gradient(90deg,#243a73,#ff7a59); }

.pv-bubble { border-radius:18px; padding:18px 22px; margin-bottom:14px; }
.pv-bubble-sys  { background:rgba(255,255,255,0.70); border:1px solid rgba(101,114,143,0.12); }
.pv-bubble-asst { background:linear-gradient(135deg,rgba(239,244,255,0.9),rgba(255,241,236,0.85)); border:1px solid rgba(92,111,164,0.15); }
.pv-bubble-title { font-size:0.68rem; text-transform:uppercase; letter-spacing:0.14em; color:#7d8ca0; font-family:'IBM Plex Mono',monospace; margin-bottom:10px; }

.pv-list { list-style:none; padding:0; margin:6px 0 0; }
.pv-list li { padding:6px 0 6px 20px; position:relative; font-size:0.88rem; color:#59697f; border-bottom:1px solid rgba(101,114,143,0.07); }
.pv-list li::before { content:"→"; position:absolute; left:0; color:#243a73; }
.pv-list-risk li::before { color:#d45c74; }
.pv-list-safe li::before { color:#2e8b6c; }

.pv-history-row { background:rgba(255,255,255,0.72); border:1px solid rgba(255,255,255,0.5); border-radius:14px; padding:14px 18px; margin-bottom:10px; }
.pv-history-top { display:flex; justify-content:space-between; font-size:0.88rem; font-weight:500; color:#1f2a3a; margin-bottom:4px; }
.pv-history-meta { font-size:0.75rem; color:#7d8ca0; }

textarea { border-radius:12px !important; border:1px solid rgba(101,114,143,0.18) !important; background:rgba(255,255,255,0.7) !important; font-size:0.95rem !important; }
.stButton > button { background:linear-gradient(135deg,#243a73,#3a5bbf) !important; color:white !important; border:none !important; border-radius:14px !important; font-weight:600 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  AWS Clients
# ─────────────────────────────────────────────
@st.cache_resource
def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name="us-east-1")

@st.cache_resource
def get_dynamodb():
    return boto3.resource("dynamodb", region_name="us-east-1")

# ─────────────────────────────────────────────
#  DynamoDB
# ─────────────────────────────────────────────
def save_to_dynamodb(news_id, original_text, ai_verdict):
    try:
        table = get_dynamodb().Table("FakeNewsHistory")
        table.put_item(Item={
            "news_id":       news_id,
            "timestamp":     datetime.utcnow().isoformat(),
            "original_text": original_text,
            "ai_verdict":    ai_verdict,
        })
        return True
    except ClientError as e:
        st.error(f"⚠️ DynamoDB error: {e.response['Error']['Message']}")
        return False
    except NoCredentialsError:
        st.error("⚠️ AWS credentials not found.")
        return False

def load_history(limit=15):
    try:
        table = get_dynamodb().Table("FakeNewsHistory")
        resp  = table.scan(Limit=limit)
        items = resp.get("Items", [])
        return sorted(items, key=lambda x: x.get("timestamp",""), reverse=True)
    except Exception:
        return []

# ─────────────────────────────────────────────
#  Bedrock
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a Forensic News Analyst expert in media literacy, cognitive bias, and investigative journalism.

Return ONLY valid JSON (no markdown fences) with this exact structure:
{
  "trust_score": <integer 0-100>,
  "fake_score": <integer 0-100>,
  "verdict": "<TRUSTWORTHY | QUESTIONABLE | MISLEADING>",
  "risk_band": "<Low Risk | Moderate Risk | High Risk>",
  "confidence": <integer 0-100>,
  "bias_analysis": "<1-2 sentences>",
  "clickbait_score": <integer 0-10>,
  "clickbait_reason": "<brief explanation>",
  "logical_issues": "<logical fallacies or gaps>",
  "red_flags": ["<flag1>", "<flag2>"],
  "positive_signals": ["<signal1>", "<signal2>"],
  "breakdown": {
    "ML Pattern Risk": <0-100>,
    "Source Credibility Risk": <0-100>,
    "Evidence Weakness Risk": <0-100>,
    "Style/Sensational Risk": <0-100>,
    "Bias/Emotion Risk": <0-100>
  },
  "verification_steps": ["<step1>", "<step2>", "<step3>"],
  "recommendation": "<one sentence>"
}
trust_score 75-100=TRUSTWORTHY, 40-74=QUESTIONABLE, 0-39=MISLEADING. fake_score = roughly 100-trust_score."""

def analyze_with_bedrock(text):
    try:
        client  = get_bedrock_client()
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "system": SYSTEM_PROMPT,
            "messages": [{"role":"user","content":f"Analyze this news content:\n\n{text}"}],
        }
        response = client.invoke_model(
            modelId="us.anthropic.claude-sonnet-4-6",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )
        body = json.loads(response["body"].read())
        raw  = body["content"][0]["text"].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        return json.loads(raw)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "ThrottlingException":
            st.error("⏳ Bedrock throttling — wait 30 seconds and retry.")
        elif code == "AccessDeniedException":
            st.error("🔒 Access denied. Check IAM role has bedrock:InvokeModel.")
        else:
            st.error(f"❌ AWS error ({code}): {e.response['Error']['Message']}")
        return None
    except NoCredentialsError:
        st.error("🔑 No AWS credentials found.")
        return None
    except Exception as e:
        st.error(f"💥 Error: {str(e)}")
        return None

# ─────────────────────────────────────────────
#  Style helpers
# ─────────────────────────────────────────────
def ring_cls(v):
    return "pv-ring-safe" if v=="TRUSTWORTHY" else "pv-ring-warn" if v=="QUESTIONABLE" else "pv-ring-risk"

def pill_cls(v):
    return "pv-pill-safe" if v=="TRUSTWORTHY" else "pv-pill-warn" if v=="QUESTIONABLE" else "pv-pill-risk"

def color(v):
    return "#2e8b6c" if v=="TRUSTWORTHY" else "#cf8e25" if v=="QUESTIONABLE" else "#d45c74"

# ─────────────────────────────────────────────
#  Load stats
# ─────────────────────────────────────────────
history_all = load_history(limit=50)
total     = len(history_all)
high_risk = sum(1 for h in history_all if "MISLEADING"  in h.get("ai_verdict",""))
low_risk  = sum(1 for h in history_all if "TRUSTWORTHY" in h.get("ai_verdict",""))
scores    = []
for h in history_all:
    try:
        v = json.loads(h.get("ai_verdict","{}"))
        scores.append(v.get("fake_score", 50))
    except Exception:
        pass
avg_fake = round(sum(scores)/len(scores)) if scores else 0

# ─────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────

# Hero
st.markdown(f"""
<div class="pv-hero">
    <div class="pv-brand">🔬 PulseVerify · Forensic Intelligence · Powered by Amazon Bedrock</div>
    <h1>Is this story real?</h1>
    <p>Paste any headline, article, or social media post. Our AI forensic engine analyzes bias, credibility, and misinformation risk in seconds.</p>
</div>
""", unsafe_allow_html=True)

# Stats
st.markdown(f"""
<div class="pv-stats">
    <div class="pv-stat"><div class="pv-stat-val">{total}</div><div class="pv-stat-lbl">Total Scans</div></div>
    <div class="pv-stat"><div class="pv-stat-val" style="color:#d45c74">{high_risk}</div><div class="pv-stat-lbl">High Risk Detected</div></div>
    <div class="pv-stat"><div class="pv-stat-val" style="color:#2e8b6c">{low_risk}</div><div class="pv-stat-lbl">Credible Stories</div></div>
    <div class="pv-stat"><div class="pv-stat-val">{avg_fake}%</div><div class="pv-stat-lbl">Avg Fake Risk</div></div>
</div>
""", unsafe_allow_html=True)

tab_scan, tab_history = st.tabs(["🔍  Analyze Story", "📜  History"])

# ── SCAN ────────────────────────────────────
with tab_scan:
    st.markdown('<div class="pv-input-card"><div class="pv-input-label">Paste your news content below</div>', unsafe_allow_html=True)
    text_input = st.text_area("news", label_visibility="collapsed",
        placeholder='"SHOCKING: Scientists discover miracle cure Big Pharma is hiding!!!"', height=150)
    c1, c2 = st.columns([1,6])
    scan_btn  = c1.button("🔍 Analyze", type="primary", use_container_width=True)
    clear_btn = c2.button("Clear")
    st.markdown('</div>', unsafe_allow_html=True)

    if clear_btn:
        st.rerun()

    if scan_btn:
        if not text_input.strip():
            st.warning("Please enter some content to analyze.")
        else:
            with st.spinner("🧠 Running forensic analysis with Claude Sonnet 4.6..."):
                result = analyze_with_bedrock(text_input.strip())

            if result:
                verdict   = result.get("verdict","QUESTIONABLE")
                fake_sc   = result.get("fake_score", 100 - result.get("trust_score",50))
                risk_band = result.get("risk_band","Moderate Risk")
                conf      = result.get("confidence",0)
                clr       = color(verdict)

                # Score ring
                st.markdown(f"""
                <div class="pv-result">
                <div class="pv-bubble-title">Forensic Analysis Result</div>
                <div class="pv-score-wrap">
                    <div class="pv-ring {ring_cls(verdict)}">
                        <div class="pv-ring-val" style="color:{clr}">{fake_sc}%</div>
                        <div class="pv-ring-lbl">fake risk</div>
                    </div>
                    <div>
                        <div class="pv-verdict">{verdict}</div>
                        <span class="pv-pill {pill_cls(verdict)}">{risk_band}</span><br/>
                        <span style="font-size:0.82rem;color:#7d8ca0;margin-top:6px;display:block">
                            Confidence: <strong>{conf}%</strong> &nbsp;|&nbsp;
                            Trust Score: <strong>{result.get("trust_score","N/A")}/100</strong>
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Breakdown bars
                breakdown = result.get("breakdown",{})
                if breakdown:
                    bars = "".join(f"""
                    <div class="pv-bar-row">
                        <div class="pv-bar-head"><span>{m}</span><span>{v}%</span></div>
                        <div class="pv-bar-track"><div class="pv-bar-fill" style="width:{v}%"></div></div>
                    </div>""" for m,v in breakdown.items())
                    st.markdown(f'<div class="pv-bubble pv-bubble-sys"><div class="pv-bubble-title">Risk Breakdown</div>{bars}</div>', unsafe_allow_html=True)

                # Red flags + positive signals
                flags   = "".join(f"<li>{f}</li>" for f in result.get("red_flags",[]))
                signals = "".join(f"<li>{s}</li>" for s in result.get("positive_signals",[]))
                st.markdown(f"""
                <div class="pv-bubble pv-bubble-asst">
                    <div class="pv-bubble-title">Editorial Signals</div>
                    <div class="pv-section-title">🚩 Red Flags</div>
                    <ul class="pv-list pv-list-risk">{flags}</ul>
                    <div class="pv-section-title">✅ Positive Signals</div>
                    <ul class="pv-list pv-list-safe">{signals}</ul>
                </div>""", unsafe_allow_html=True)

                # Bias + logical issues
                st.markdown(f"""
                <div class="pv-bubble pv-bubble-sys">
                    <div class="pv-bubble-title">Detailed Analysis</div>
                    <div class="pv-section-title">Bias Analysis</div>
                    <p style="font-size:0.88rem;color:#59697f;margin:0 0 12px">{result.get("bias_analysis","N/A")}</p>
                    <div class="pv-section-title">Logical Issues</div>
                    <p style="font-size:0.88rem;color:#59697f;margin:0 0 12px">{result.get("logical_issues","None detected")}</p>
                    <div class="pv-section-title">Clickbait Score: {result.get("clickbait_score",0)}/10</div>
                    <p style="font-size:0.88rem;color:#59697f;margin:0">{result.get("clickbait_reason","N/A")}</p>
                </div>""", unsafe_allow_html=True)

                # Verification steps
                steps = "".join(f"<li>{s}</li>" for s in result.get("verification_steps",[]))
                st.markdown(f"""
                <div class="pv-bubble pv-bubble-asst">
                    <div class="pv-bubble-title">How to Verify This Story</div>
                    <ul class="pv-list">{steps}</ul>
                    <div class="pv-section-title" style="margin-top:14px">Recommendation</div>
                    <p style="font-size:0.88rem;color:#59697f;margin:0">{result.get("recommendation","")}</p>
                </div>
                </div>""", unsafe_allow_html=True)

                # Save to DynamoDB
                news_id = str(uuid.uuid4())
                if save_to_dynamodb(news_id, text_input.strip(), json.dumps(result)):
                    st.caption(f"✅ Saved to DynamoDB — ID: `{news_id}`")

# ── HISTORY ─────────────────────────────────
with tab_history:
    st.markdown("#### 📜 Recent Analyses")
    if st.button("🔄 Refresh"):
        st.rerun()
    records = load_history(limit=15)
    if not records:
        st.info("No history yet. Run your first scan!")
    else:
        for rec in records:
            try:
                v       = json.loads(rec.get("ai_verdict","{}"))
                verdict = v.get("verdict","UNKNOWN")
                fake_sc = v.get("fake_score","?")
                conf    = v.get("confidence","?")
                rb      = v.get("risk_band","")
                ts      = rec.get("timestamp","")[:19].replace("T"," ")
                txt     = rec.get("original_text","")[:120]
                clr     = color(verdict)
                st.markdown(f"""
                <div class="pv-history-row">
                    <div class="pv-history-top">
                        <span>{txt}{'…' if len(rec.get('original_text',''))>120 else ''}</span>
                        <span style="color:{clr};font-weight:700;font-family:'IBM Plex Mono',monospace">{fake_sc}%</span>
                    </div>
                    <div class="pv-history-meta">
                        <span style="color:{clr};font-weight:600">{verdict}</span> | {rb} | Confidence {conf}% | {ts}
                    </div>
                </div>""", unsafe_allow_html=True)
            except Exception:
                pass
