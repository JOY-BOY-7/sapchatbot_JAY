import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import re
import difflib
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
import matplotlib.pyplot as plt
import io
import contextlib
import traceback

st.set_page_config(page_title="SAP Odata ChatBot", layout="wide")

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
# OData XML Parsing
# -----------------------------
def parse_odata_xml(xml_text):
    ns = {
        'atom': 'http://www.w3.org/2005/Atom',
        'm': 'http://schemas.microsoft.com/ado/2007/08/dataservices/metadata',
        'd': 'http://schemas.microsoft.com/ado/2007/08/dataservices'
    }
    root = ET.fromstring(xml_text)
    entries = root.findall('.//atom:entry', ns)
    data = []

    for entry in entries:
        props = entry.find('.//m:properties', ns)
        if props is None:
            continue
        record = {}
        for child in props:
            tag = re.sub(r'^{.*}', '', child.tag)
            record[tag] = child.text
        data.append(record)

    if not data:
        raise ValueError("No valid data entries found in OData response.")
    return pd.DataFrame(data)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🤖 SAP Odata / CSV / Excel ChatBot")

with st.sidebar:
    st.header("🧠 Gemini Setup")
    gemini_url = st.text_input(
        "REST Endpoint URL",
        value="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
    )
    gemini_key = st.text_input("Gemini API Key", type="password")
    timeout = st.number_input("Timeout (sec)", value=30, min_value=5, max_value=120)

    st.header("📌 Data Source")
    source = st.selectbox("Select Data Source", ["OData", "CSV", "Excel"])

    if source == "OData":
        odata_url = st.text_input("OData Service URL (EntitySet)")
        username = st.text_input("Username (optional)")
        password = st.text_input("Password (optional)", type="password")

    if source == "CSV":
        csv_file = st.file_uploader("Upload CSV", type=["csv"])

    if source == "Excel":
        excel_file = st.file_uploader("Upload Excel", type=["xlsx", "xls"])

if not gemini_url or not gemini_key:
    st.warning("Please enter your Gemini endpoint and API key.")
    st.stop()

# -----------------------------
# Load Data According to Source
# -----------------------------
if source == "OData":
    if not odata_url:
        st.info("Enter OData URL.")
        st.stop()

    auth = (username, password) if username and password else None
    resp = requests.get(odata_url, auth=auth, headers={"Accept": "application/atom+xml"}, timeout=timeout)
    if resp.status_code != 200:
        st.error("OData fetch failed!")
        st.text(resp.text)
        st.stop()
    df = parse_odata_xml(resp.text)

elif source == "CSV":
    if not csv_file:
        st.info("Upload a CSV file.")
        st.stop()
    df = pd.read_csv(csv_file)

elif source == "Excel":
    if not excel_file:
        st.info("Upload an Excel file.")
        st.stop()
    df = pd.read_excel(excel_file)

# -----------------------------
# Normalize & prep DataFrame
# -----------------------------
orig_cols = df.columns.tolist()
norm_map = {c: normalize_col(c) for c in orig_cols}
df.columns = [norm_map[c] for c in orig_cols]
fuzzy_map = fuzzy_column_map(df.columns)

for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors='ignore')

st.success(f"✅ Loaded {len(df)} rows.")
st.dataframe(df.head(100))

# -----------------------------
# Prepare Prompt
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
7. When grouping numeric columns, use aggregation (sum, mean, count)
8. When a name came up keep in mind that its not full name only part of name
9. Do not answer general knowledge questions (outside dataset); reply with "only ask questions related to data please".
10. Always handle NaN values safely:
   - For string filters: use str.contains(..., na=False)
   - For numeric operations: safely handle empty sequences
11. i will give you an example understand it and dynamically answer that of questions
example 1: user question:total revenue for GST?
gemini generated code:df[df.select_dtypes(include=['object','string']).apply(lambda col: col.str.contains('gst', case=False, na=False)).any(axis=1)]['gross_amt'].sum()
example 2:What was the revenue under "EAP and Other Grants" in April FY 2024-25?
gemini generated code:df[(df['"search all columns of cat"'].str.contains('EAP and Other Grants', case=False, na=False)) & (df['month'] == 'Apr') & (df['fin_year'] == 'FY 2024-25')]['gross_amt'].sum()
"""

# -----------------------------
# User Question
# -----------------------------
user_q = st.text_input("Ask your question:")
if not user_q:
    st.stop()

with st.spinner("💡 Thinking..."):
    resp = call_gemini_json(gemini_url, gemini_key, PROMPT_PANDAS_TRANSLATE + "\nQuestion: " + user_q, timeout)
    js = extract_json_from_response(resp)

if not js or "expr" not in js:
    st.warning("Could not parse expression.")
    st.json(resp)
    st.stop()

expr = js["expr"]
explain = js.get("explain", "")

# Execute
result = safe_exec(expr, df)

# Display
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
        st.write(result)

# Explanation
PROMPT_ENGLISH = f"""
Question: {user_q}
Result: {repr(result)}
Explain clearly.
"""
resp2 = call_gemini_json(gemini_url, gemini_key, PROMPT_ENGLISH, timeout)
try:
    text = resp2["candidates"][0]["content"]["parts"][0]["text"]
except:
    text = str(resp2)

st.markdown("### 💬 Chatbot Answer")
st.write(text)

st.markdown("---")
st.code(expr)
st.caption(explain)
