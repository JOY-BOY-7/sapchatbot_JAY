import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import re
import difflib
import matplotlib.pyplot as plt
import io
import contextlib
import traceback

st.set_page_config(page_title="CSV Data ChatBot", layout="wide")

# -----------------------------
# Utility Functions
# -----------------------------
def normalize_col(c):
    return re.sub(r"[^0-9a-z_]", "_", c.strip().lower())

def fuzzy_column_map(columns):
    mapping = {}
    for c in columns:
        mapping[c.lower()] = c
        for token in c.lower().split("_"):
            mapping[token] = c
    return mapping

def extract_json_from_response(resp):
    try:
        if isinstance(resp, dict):
            if "candidates" in resp:
                text = resp["candidates"][0]["content"]["parts"][0]["text"]
            elif "text" in resp:
                text = resp["text"]
            else:
                text = json.dumps(resp)
        else:
            text = str(resp)
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except:
        pass
    return None

def fuzzy_filter(df, col, value):
    col_values = df[col].dropna().astype(str).unique()
    closest = difflib.get_close_matches(str(value), col_values, n=1, cutoff=0.6)
    if closest:
        return df[df[col].fillna('').str.contains(closest[0], case=False, na=False)]
    else:
        return df[df[col].fillna('').str.contains(str(value), case=False, na=False)]

# -----------------------------
# Safe Exec
# -----------------------------
def safe_exec(expr, df):
    local_env = {"df": df, "pd": pd, "np": np, "plt": plt, "re": re, "fuzzy_filter": fuzzy_filter}

    with st.expander("🧠 Gemini Generated Python Code", expanded=False):
        st.code(expr, language="python")

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        try:
            try:
                result = eval(expr, {}, local_env)
            except:
                exec(expr, {}, local_env)
                for k, v in reversed(local_env.items()):
                    if isinstance(v, (pd.DataFrame, pd.Series, plt.Figure)):
                        result = v
                        break
                else:
                    result = "✅ Code executed successfully"
        except Exception:
            st.error(f"⚠️ Error executing expression:\n\n{traceback.format_exc()}")
            return None
    return result

# -----------------------------
# Gemini REST API Call
# -----------------------------
def call_gemini_json(url, key, prompt, timeout=40):
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        return r.json()
    except:
        return {"text": r.text}

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🤖 CSV Data ChatBot")

with st.sidebar:
    st.header("🧠 Gemini Setup")
    gemini_url = st.text_input(
        "REST Endpoint URL",
        value="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
    )
    gemini_key = st.text_input("Gemini API Key", type="password")
    timeout = st.number_input("Timeout (sec)", value=30, min_value=5, max_value=120)

    st.header("📁 CSV Upload")
    file = st.file_uploader("Upload CSV", type=["csv"])

if not gemini_url or not gemini_key:
    st.warning("Provide Gemini endpoint and key.")
    st.stop()

if not file:
    st.info("Upload a CSV file to continue.")
    st.stop()

# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv(file)

# Normalize column names
orig_cols = df.columns.tolist()
norm_map = {c: normalize_col(c) for c in orig_cols}
df.columns = [norm_map[c] for c in orig_cols]
reverse_map = {v: k for k, v in norm_map.items()}
fuzzy_map = fuzzy_column_map(df.columns)

for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors='ignore')

st.success(f"✅ Loaded {len(df)} rows.")
st.dataframe(df.head(100))

# -----------------------------
# Schema
# -----------------------------
schema = []
for c in df.columns:
    sample = str(df[c].dropna().iloc[0]) if df[c].dropna().shape[0] > 0 else ""
    schema.append({"name": c, "dtype": str(df[c].dtype), "sample": sample})
schema_json = json.dumps(schema, indent=2)
aliases = ", ".join(list(fuzzy_map.keys()))

PROMPT_PANDAS_TRANSLATE = f"""
You are an expert data reasoning assistant.
DataFrame 'df' schema:
{schema_json}

Column aliases (fuzzy matches allowed): {aliases}

Return ONLY JSON:
  "explain": brief description
  "expr": valid pandas one-liner
"""

# -----------------------------
# User Question
# -----------------------------
user_q = st.text_input("Ask something about your CSV data:")
if not user_q:
    st.stop()

# Step 1: get expression
with st.spinner("💡 Thinking with Gemini..."):
    resp = call_gemini_json(gemini_url, gemini_key, PROMPT_PANDAS_TRANSLATE + "\nQuestion: " + user_q, timeout)
    js = extract_json_from_response(resp)

if not js or "expr" not in js:
    st.warning("⚠️ Could not parse expression from LLM.")
    st.json(resp)
    st.stop()

expr = js["expr"]
explain = js.get("explain", "")

# Step 2: Execute
result = safe_exec(expr, df)

# Step 3: Display
if isinstance(result, pd.DataFrame):
    st.dataframe(result)
elif isinstance(result, pd.Series):
    st.dataframe(result.to_frame())
else:
    fig = plt.gcf()
    if fig.get_axes():
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.write(f"✅ Result: {result}")

# Step 4: Final English answer
PROMPT_ENGLISH = f"""
Question: {user_q}
Result: {repr(result)}
Explain the answer clearly.
"""

with st.spinner("🗣️ Explaining..."):
    resp2 = call_gemini_json(gemini_url, gemini_key, PROMPT_ENGLISH, timeout)
    try:
        text = resp2["candidates"][0]["content"]["parts"][0]["text"]
    except:
        text = str(resp2)

st.markdown("### 💬 Chatbot Answer")
st.write(text)

st.markdown("---")
st.code(expr, language="python")
st.caption(f"Explanation: {explain}")
