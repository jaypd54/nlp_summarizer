import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import time
import re

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="T5 Summarizer",
    page_icon="✦",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0e0e10;
    color: #e8e4dc;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.5rem 2rem 4rem; max-width: 780px; }

/* ── Hero title ── */
.hero {
    text-align: center;
    padding: 2.8rem 0 2rem;
}
.hero-glyph {
    font-size: 2.2rem;
    color: #c8b97a;
    letter-spacing: 0.3em;
    display: block;
    margin-bottom: 0.5rem;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    font-weight: 400;
    letter-spacing: -0.01em;
    line-height: 1.1;
    margin: 0;
    background: linear-gradient(135deg, #e8e4dc 40%, #c8b97a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    font-weight: 300;
    color: #6e6a60;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.7rem;
}

/* ── Divider ── */
.thin-rule {
    border: none;
    border-top: 1px solid #2a2a2e;
    margin: 0.5rem 0 2rem;
}

/* ── Labels ── */
.field-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #6e6a60;
    margin-bottom: 0.5rem;
}

/* ── Textarea ── */
textarea {
    background: #16161a !important;
    border: 1px solid #2a2a2e !important;
    border-radius: 6px !important;
    color: #e8e4dc !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.65 !important;
    caret-color: #c8b97a !important;
    transition: border-color 0.2s ease !important;
    resize: vertical !important;
}
textarea:focus {
    border-color: #c8b97a !important;
    box-shadow: 0 0 0 2px rgba(200,185,122,0.12) !important;
    outline: none !important;
}
textarea::placeholder { color: #3e3e44 !important; }

/* ── Sliders ── */
.stSlider > div > div > div > div {
    background: #c8b97a !important;
}
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: #c8b97a !important;
    border-color: #c8b97a !important;
}

/* ── Button ── */
.stButton > button {
    width: 100%;
    background: #c8b97a !important;
    color: #0e0e10 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.75rem 1.5rem !important;
    cursor: pointer !important;
    transition: opacity 0.15s ease, transform 0.1s ease !important;
    margin-top: 0.25rem;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Result card ── */
.result-card {
    background: #13131a;
    border: 1px solid #2a2a2e;
    border-left: 3px solid #c8b97a;
    border-radius: 6px;
    padding: 1.4rem 1.6rem;
    margin-top: 1.8rem;
    animation: fadeSlide 0.4s ease both;
}
@keyframes fadeSlide {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #c8b97a;
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.result-text {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    line-height: 1.7;
    color: #e8e4dc;
}
.stats-row {
    display: flex;
    gap: 1.5rem;
    margin-top: 1.2rem;
    padding-top: 1rem;
    border-top: 1px solid #2a2a2e;
}
.stat {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}
.stat-val {
    font-family: 'DM Mono', monospace;
    font-size: 1rem;
    font-weight: 500;
    color: #c8b97a;
}
.stat-key {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4a4a50;
}

/* ── Settings expander ── */
details summary {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #6e6a60 !important;
    cursor: pointer !important;
}
.streamlit-expanderHeader {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    color: #6e6a60 !important;
    background: transparent !important;
    border: 1px solid #2a2a2e !important;
    border-radius: 4px !important;
}
.streamlit-expanderContent {
    background: #13131a !important;
    border: 1px solid #2a2a2e !important;
    border-top: none !important;
    border-radius: 0 0 4px 4px !important;
    padding: 1rem 1.2rem !important;
}

/* ── Error ── */
.err-box {
    background: #1e1014;
    border: 1px solid #5a2030;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #e07080;
}

/* ── Word count chip ── */
.wc-chip {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #4a4a50;
    text-align: right;
    margin-top: 0.3rem;
    letter-spacing: 0.1em;
}
</style>
""", unsafe_allow_html=True)


# ── Model loader ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model_name = "t5-small"

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    model.eval()
    return tokenizer, model


def generate_summary(text: str, max_len: int, min_len: int, num_beams: int,
                     rep_penalty: float, tokenizer, model) -> str:
    input_text = "summarize: " + text.strip()
    inputs = tokenizer(input_text, max_length=512, truncation=True, return_tensors="pt")
    ids = model.generate(
        inputs["input_ids"],
        max_length=max_len,
        min_length=min_len,
        num_beams=num_beams,
        repetition_penalty=rep_penalty,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def compression_ratio(original: str, summary: str) -> str:
    orig_wc = word_count(original)
    summ_wc = word_count(summary)
    if orig_wc == 0:
        return "—"
    ratio = round((1 - summ_wc / orig_wc) * 100)
    return f"{ratio}%"


# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <span class="hero-glyph">✦ ✦ ✦</span>
  <h1>Summarizer</h1>
  <p class="hero-sub">T5 · Abstractive · Seq2Seq</p>
</div>
<hr class="thin-rule"/>
""", unsafe_allow_html=True)

# Model path input (sidebar)
with st.sidebar:
    st.markdown('<p class="field-label">Model path</p>', unsafe_allow_html=True)
    st.markdown('<hr class="thin-rule"/>', unsafe_allow_html=True)
    st.markdown('<p class="field-label">About</p>', unsafe_allow_html=True)
    st.caption("T5-small fine-tuned for abstractive summarization. Paste any article or passage to generate a concise summary.")

# Load model
try:
    with st.spinner("Loading model weights…"):
        tokenizer, model = load_model()
    model_ready = True
except Exception as e:
    model_ready = False
    st.markdown(f'<div class="err-box">⚠ Could not load model from <code>{model_path}</code><br>{e}</div>', unsafe_allow_html=True)

# Text input
st.markdown('<p class="field-label">Input text</p>', unsafe_allow_html=True)
input_text = st.text_area(
    label="input",
    label_visibility="collapsed",
    placeholder="Paste an article, paragraph, or document here…",
    height=220,
)

wc = word_count(input_text)
st.markdown(f'<p class="wc-chip">{wc} words</p>', unsafe_allow_html=True)

# Advanced settings
with st.expander("Generation settings"):
    col1, col2 = st.columns(2)
    with col1:
        max_len = st.slider("Max length", 40, 200, 80, 5)
        num_beams = st.slider("Beam width", 2, 10, 6)
    with col2:
        min_len = st.slider("Min length", 10, 80, 20, 5)
        rep_penalty = st.slider("Repetition penalty", 1.0, 2.5, 1.5, 0.1)

# Summarize button
summarize = st.button("✦  Summarize", disabled=not model_ready)

# Output
if summarize:
    if not input_text.strip():
        st.markdown('<div class="err-box">Please enter some text to summarize.</div>', unsafe_allow_html=True)
    elif wc < 20:
        st.markdown('<div class="err-box">Text is too short — try at least 20 words for a meaningful summary.</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Generating summary…"):
            t0 = time.perf_counter()
            summary = generate_summary(
                input_text, max_len, min_len, num_beams, rep_penalty, tokenizer, model
            )
            elapsed = time.perf_counter() - t0

        st.markdown(f"""
        <div class="result-card">
          <div class="result-header">
            <span>✦</span> Summary
          </div>
          <div class="result-text">{summary}</div>
          <div class="stats-row">
            <div class="stat">
              <span class="stat-val">{word_count(summary)}</span>
              <span class="stat-key">words</span>
            </div>
            <div class="stat">
              <span class="stat-val">{compression_ratio(input_text, summary)}</span>
              <span class="stat-key">compressed</span>
            </div>
            <div class="stat">
              <span class="stat-val">{elapsed:.2f}s</span>
              <span class="stat-key">inference</span>
            </div>
            <div class="stat">
              <span class="stat-val">{num_beams}</span>
              <span class="stat-key">beams</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
