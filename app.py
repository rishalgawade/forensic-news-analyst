import streamlit as st
import boto3
import json
import uuid
from datetime import datetime
from botocore.exceptions import ClientError, NoCredentialsError

# ─────────────────────────────────────────────
#  Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Forensic News Analyst",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0e1117; }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid #30475e;
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .hero h1 { color: #e2e8f0; font-size: 2.4rem; margin: 0; }
    .hero p  { color: #94a3b8; font-size: 1rem; margin-top: 0.5rem; }

    /* Trust meter colours */
    .trust-high   { background: linear-gradient(135deg,#064e3b,#065f46); border:1px solid #10b981; color:#6ee7b7; }
    .trust-medium { background: linear-gradient(135deg,#78350f,#92400e); border:1px solid #f59e0b; color:#fcd34d; }
    .trust-low    { background: linear-gradient(135deg,#7f1d1d,#991b1b); border:1px solid #ef4444; color:#fca5a5; }

    .verdict-box {
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin: 1.5rem 0;
    }
    .verdict-title { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.4rem; }
    .verdict-sub   { font-size: 0.85rem; opacity: 0.75; }

    /* Analysis card */
    .analysis-card {
        background: #1e2433;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .analysis-card h4 { color: #60a5fa; margin-top: 0; }

    /* History row */
    .history-item {
        background: #1a2030;
        border-left: 4px solid #3b82f6;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        font-size: 0.85rem;
        color: #cbd5e1;
    }

    /* Sidebar */
    .sidebar-stat {
        background: #1e2433;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sidebar-stat .num { font-size: 1.8rem; font-weight: 700; color: #60a5fa; }
    .sidebar-stat .lbl { font-size: 0.75rem; color: #64748b; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  AWS Clients  (cached so they're created once)
# ─────────────────────────────────────────────
@st.cache_resource
def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name="us-east-1")

@st.cache_resource
def get_dynamodb():
    return boto3.resource("dynamodb", region_name="us-east-1")

# ─────────────────────────────────────────────
#  DynamoDB helpers
# ─────────────────────────────────────────────
def save_to_dynamodb(news_id: str, original_text: str, ai_verdict: str) -> bool:
    """Persist one analysis record; returns True on success."""
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
        st.error("⚠️ AWS credentials not found. Check your IAM role or environment variables.")
        return False


def load_history(limit: int = 10) -> list:
    """Return the most-recent records from FakeNewsHistory."""
    try:
        table = get_dynamodb().Table("FakeNewsHistory")
        resp  = table.scan(Limit=limit)
        items = resp.get("Items", [])
        return sorted(items, key=lambda x: x.get("timestamp", ""), reverse=True)
    except Exception:
        return []

# ─────────────────────────────────────────────
#  Bedrock / Claude call
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a Forensic News Analyst — an expert in media literacy, cognitive bias, and investigative journalism.

Your task is to evaluate the news headline or article excerpt provided by the user.

Return your analysis ONLY as valid JSON (no markdown fences, no extra text) with this exact structure:
{
  "trust_score": <integer 0-100>,
  "verdict": "<TRUSTWORTHY | QUESTIONABLE | MISLEADING>",
  "bias_analysis": "<1-2 sentence analysis of political/emotional bias>",
  "clickbait_score": <integer 0-10>,
  "clickbait_reason": "<brief explanation>",
  "logical_issues": "<any logical fallacies or factual gaps>",
  "red_flags": ["<flag1>", "<flag2>"],
  "recommendation": "<one-sentence advice for the reader>"
}

Scoring guide:
- trust_score 75-100 → TRUSTWORTHY  (factual, neutral language, verifiable)
- trust_score 40-74  → QUESTIONABLE (some bias or unverified claims)
- trust_score 0-39   → MISLEADING   (heavy bias, logical fallacies, or clear misinformation)
"""

def analyze_with_bedrock(headline: str) -> dict | None:
    """Call Claude 3.7 Sonnet on Bedrock; return parsed JSON dict or None."""
    try:
        client  = get_bedrock_client()
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": f"Analyze this news content:\n\n{headline}"}
            ],
        }
        response = client.invoke_model(
            modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )
        body = json.loads(response["body"].read())
        raw  = body["content"][0]["text"].strip()

        # Strip accidental markdown fences just in case
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)

    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "ThrottlingException":
            st.error("⏳ Bedrock is throttling requests. Please wait 30 seconds and try again.")
        elif code == "AccessDeniedException":
            st.error("🔒 Access denied to Bedrock. Verify your IAM role has `bedrock:InvokeModel` permission.")
        else:
            st.error(f"❌ AWS error ({code}): {e.response['Error']['Message']}")
        return None

    except NoCredentialsError:
        st.error("🔑 No AWS credentials found. Ensure the App Runner instance role is attached.")
        return None

    except json.JSONDecodeError as e:
        st.error(f"🔧 Failed to parse AI response as JSON: {e}")
        return None

    except Exception as e:
        st.error(f"💥 Unexpected error: {str(e)}")
        return None

# ─────────────────────────────────────────────
#  Trust Meter UI helper
# ─────────────────────────────────────────────
def render_trust_meter(score: int, verdict: str):
    css_class = (
        "trust-high"   if score >= 75 else
        "trust-medium" if score >= 40 else
        "trust-low"
    )
    emoji = "✅" if score >= 75 else "⚠️" if score >= 40 else "🚨"
    bar_color = "#10b981" if score >= 75 else "#f59e0b" if score >= 40 else "#ef4444"

    st.markdown(f"""
    <div class="verdict-box {css_class}">
        <div class="verdict-title">{emoji} {verdict}</div>
        <div class="verdict-sub">Trust Score: {score} / 100</div>
    </div>
    """, unsafe_allow_html=True)

    # Native progress bar (Streamlit doesn't support coloured bars natively,
    # so we use HTML for the visual bar)
    st.markdown(f"""
    <div style="background:#2d3748;border-radius:8px;height:14px;overflow:hidden;margin-bottom:1rem">
        <div style="width:{score}%;background:{bar_color};height:100%;border-radius:8px;
                    transition:width .6s ease;"></div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 Forensic News Analyst")
    st.markdown("---")
    st.markdown("**How It Works**")
    st.caption("1. Paste a headline or excerpt\n2. Click **Scan**\n3. Receive an AI-powered forensic report")
    st.markdown("---")
    st.markdown("**AWS Stack**")
    for svc, icon in [("Bedrock (Claude 3.7)", "🤖"), ("DynamoDB", "🗄️"), ("App Runner", "🚀")]:
        st.markdown(f"{icon} `{svc}`")
    st.markdown("---")

    history = load_history(limit=20)
    total   = len(history)
    trusty  = sum(1 for h in history if "TRUSTWORTHY" in h.get("ai_verdict",""))
    mislead = sum(1 for h in history if "MISLEADING"  in h.get("ai_verdict",""))

    col1, col2, col3 = st.columns(3)
    col1.markdown(f'<div class="sidebar-stat"><div class="num">{total}</div><div class="lbl">Total</div></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="sidebar-stat"><div class="num" style="color:#10b981">{trusty}</div><div class="lbl">Trusted</div></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="sidebar-stat"><div class="num" style="color:#ef4444">{mislead}</div><div class="lbl">Misleading</div></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Main Page
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🔬 Forensic News Analyst</h1>
    <p>AI-powered detection of bias, clickbait & misinformation — powered by Claude 3.7 Sonnet on AWS Bedrock</p>
</div>
""", unsafe_allow_html=True)

tab_scan, tab_history = st.tabs(["📡  Scan", "📜  History"])

# ── SCAN TAB ──────────────────────────────────
with tab_scan:
    headline_input = st.text_area(
        label="Paste your headline or article excerpt below:",
        placeholder='e.g. "Scientists SHOCKED: This common food DESTROYS your brain in 7 days!"',
        height=140,
        label_visibility="visible",
    )

    col_btn, col_clear = st.columns([1, 5])
    scan_clicked  = col_btn.button("🔍 Scan", type="primary", use_container_width=True)
    clear_clicked = col_clear.button("Clear", use_container_width=False)

    if clear_clicked:
        st.rerun()

    if scan_clicked:
        if not headline_input.strip():
            st.warning("Please enter some text before scanning.")
        else:
            with st.spinner("🧠 Analysing with Claude 3.7 Sonnet…"):
                result = analyze_with_bedrock(headline_input.strip())

            if result:
                # ── Trust Meter ──
                render_trust_meter(result["trust_score"], result["verdict"])

                # ── Detail Cards ──
                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                c1, c2 = st.columns(2)

                with c1:
                    st.markdown("#### 🎯 Bias Analysis")
                    st.info(result.get("bias_analysis", "N/A"))
                    st.markdown("#### 🧩 Logical Issues")
                    st.warning(result.get("logical_issues", "None detected"))

                with c2:
                    cb = result.get("clickbait_score", 0)
                    st.markdown(f"#### 🎣 Clickbait Score: **{cb}/10**")
                    st.progress(cb / 10)
                    st.caption(result.get("clickbait_reason", ""))
                    st.markdown("#### 🚩 Red Flags")
                    flags = result.get("red_flags", [])
                    if flags:
                        for f in flags:
                            st.error(f"• {f}")
                    else:
                        st.success("No significant red flags detected.")

                st.markdown("#### 💡 Recommendation")
                st.success(result.get("recommendation", ""))
                st.markdown("</div>", unsafe_allow_html=True)

                # ── Save to DynamoDB ──
                news_id     = str(uuid.uuid4())
                verdict_str = json.dumps(result)
                saved = save_to_dynamodb(news_id, headline_input.strip(), verdict_str)
                if saved:
                    st.caption(f"✅ Saved to DynamoDB — ID: `{news_id}`")

# ── HISTORY TAB ───────────────────────────────
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
                v = json.loads(rec.get("ai_verdict", "{}"))
                score   = v.get("trust_score", "?")
                verdict = v.get("verdict", "UNKNOWN")
                color   = "#10b981" if verdict == "TRUSTWORTHY" else "#f59e0b" if verdict == "QUESTIONABLE" else "#ef4444"
                ts      = rec.get("timestamp","")[:19].replace("T"," ")
                text    = rec.get("original_text","")[:120]
                st.markdown(f"""
                <div class="history-item">
                    <span style="color:{color};font-weight:700">{verdict} ({score}/100)</span>
                    &nbsp;|&nbsp; <span style="color:#64748b">{ts}</span><br/>
                    {text}{'…' if len(rec.get('original_text',''))>120 else ''}
                </div>
                """, unsafe_allow_html=True)
            except Exception:
                st.caption(f"⚠️ Could not parse record `{rec.get('news_id','?')}`")
