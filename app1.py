import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import re
import difflib
from hdbcli import dbapi  # SAP HANA DB client

st.set_page_config(page_title="Gemini 2.5 Flash Lite Chatbot — HANA DB", layout="wide")

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

def validate_expr(expr):
    forbidden = ["import", "exec", "eval", "subprocess", "os.", "sys."]
    if any(f in expr for f in forbidden):
        raise ValueError("Unsafe code detected.")
    return True

def fuzzy_filter(df, col, value):
    col_values = df[col].dropna().astype(str).unique()
    closest = difflib.get_close_matches(str(value), col_values, n=1, cutoff=0.6)
    if closest:
        return df[df[col] == closest[0]]
    else:
        return df[df[col] == value]

def safe_exec(expr, df):
    safe_globals = {"pd": pd, "np": np, "fuzzy_filter": fuzzy_filter}
    expr_fixed = re.sub(
        r"df\[['\"](\w+)['\"]\]\s*==\s*['\"]([^'\"]+)['\"]",
        r"fuzzy_filter(df, '\1', '\2')",
        expr
    )
    try:
        return eval(expr_fixed, safe_globals, {"df": df})
    except Exception:
        return eval(expr, safe_globals, {"df": df})

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
st.title("🤖 Gemini 2.5 Flash Lite — SAP HANA Chatbot (Query Enabled)")

with st.sidebar:
    st.header("🧠 Gemini Setup")
    gemini_url = st.text_input(
        "REST Endpoint URL",
        value="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
    )
    gemini_key = st.text_input("Gemini API Key (AIza...)", type="password")
    timeout = st.number_input("Timeout (sec)", value=30, min_value=5, max_value=120)

    st.header("🖥 SAP HANA Connection")
    hana_host = st.text_input("Host")
    hana_port = st.number_input("Port", value=30015)
    hana_user = st.text_input("Username")
    hana_password = st.text_input("Password", type="password")
    hana_query = st.text_area("SQL Query", placeholder="SELECT * FROM MY_TABLE WHERE ROWNUM <= 100")

    # --- Connection Test ---
    if st.button("🔌 Test Connection"):
        if not hana_host or not hana_user or not hana_password:
            st.warning("Please enter host, username, and password to test connection.")
        else:
            try:
                conn = dbapi.connect(
                    address=hana_host,
                    port=hana_port,
                    user=hana_user,
                    password=hana_password
                )
                conn.close()
                st.success("✅ Connection successful!")
            except Exception as e:
                st.error(f"❌ Connection failed: {e}")

# -----------------------------
# Validate Gemini setup
# -----------------------------
if not gemini_url or not gemini_key:
    st.warning("Please enter your Gemini URL and API key.")
    st.stop()

# -----------------------------
# Validate HANA inputs
# -----------------------------
if not hana_host or not hana_user or not hana_password or not hana_query:
    st.info("Enter SAP HANA connection details and SQL query.")
    st.stop()

# -----------------------------
# Fetch Data from SAP HANA
# -----------------------------
try:
    conn = dbapi.connect(
        address=hana_host,
        port=hana_port,
        user=hana_user,
        password=hana_password
    )
    df = pd.read_sql(hana_query, conn)
    conn.close()
except Exception as e:
    st.error(f"❌ Failed to fetch data from SAP HANA: {e}")
    st.stop()

# -----------------------------
# Prepare DataFrame
# -----------------------------
orig_cols = df.columns.tolist()
norm_map = {c: normalize_col(c) for c in orig_cols}
df.columns = [norm_map[c] for c in orig_cols]
reverse_map = {v: k for k, v in norm_map.items()}
fuzzy_map = fuzzy_column_map(df.columns)

st.success(f"✅ Loaded {len(df)} rows from SAP HANA.")
st.dataframe(df.head(100))

# -----------------------------
# Prepare Gemini Prompt
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

Rules:
1. Use closest matching column names
2. String comparisons are case-insensitive and fuzzy (handled automatically)
3. Numeric operations safe
4. Never hallucinate columns/values
5. No loops/imports/prints
6. Always valid Python one-liner
"""

# -----------------------------
# User Question
# -----------------------------
user_q = st.text_input("Ask your question about this SAP HANA data:")
if not user_q:
    st.stop()

# Step 1: Generate pandas expression
with st.spinner("💡 Thinking with Gemini..."):
    resp = call_gemini_json(gemini_url, gemini_key, PROMPT_PANDAS_TRANSLATE + "\nQuestion: " + user_q, timeout)
    js = extract_json_from_response(resp)

if not js or "expr" not in js:
    st.error("❌ Gemini response parsing failed:")
    st.json(resp)
    st.stop()

expr = js["expr"]
explain = js.get("explain", "")

# Step 2: Execute safely
try:
    validate_expr(expr)
    result = safe_exec(expr, df)
except Exception as e:
    st.error(f"⚠️ Error executing expression: {e}")
    st.code(expr)
    st.stop()

# Step 3: Display result
if isinstance(result, pd.DataFrame):
    st.markdown("### 📈 Result Table")
    st.dataframe(result)
elif isinstance(result, pd.Series):
    st.markdown("### 📊 Result Series")
    st.dataframe(result.to_frame())
else:
    st.markdown(f"### ✅ Result: **{result}**")

# Step 4: English explanation
PROMPT_ENGLISH = f"""
You are a helpful assistant. 
Question: {user_q}
The result is: {repr(result)}
Give the **answer with explanation**, in natural English.
"""
with st.spinner("🗣️ Generating natural language answer..."):
    resp2 = call_gemini_json(gemini_url, gemini_key, PROMPT_ENGLISH, timeout)
    try:
        text = resp2["candidates"][0]["content"]["parts"][0]["text"]
    except:
        text = str(resp2)

st.markdown("### 💬 Chatbot Answer")
st.write(text)

st.markdown("---")
st.markdown("### 🧾 Executed Expression")
st.code(expr, language="python")
st.caption(f"Explanation: {explain}")
