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
import ssl
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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

def validate_expr(expr):
    forbidden = ["subprocess", "os.", "sys.", "open(", "eval(", "exec(", "__import__", "input(", "print("]
    if any(f in expr for f in forbidden):
        raise ValueError("Unsafe code detected.")
    return True

def fuzzy_filter(df, col, value):
    col_values = df[col].dropna().astype(str).unique()
    closest = difflib.get_close_matches(str(value), col_values, n=1, cutoff=0.6)
    if closest:
        return df[df[col].fillna('').str.contains(closest[0], case=False, na=False)]
    else:
        return df[df[col].fillna('').str.contains(str(value), case=False, na=False)]

# -----------------------------
# Safe Execution (for Gemini Code)
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
                    result = "✅ Code executed successfully (no direct result returned)"
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
# SSL Adapter (fixed for legacy SAP TLS + verify=False)
# -----------------------------
class SSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.options |= 0x4  # SSL_OP_LEGACY_SERVER_CONNECT
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        kwargs['ssl_context'] = ctx
        return super(SSLAdapter, self).init_poolmanager(*args, **kwargs)

# -----------------------------
# Cached OData Fetch
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_odata_data(odata_url, username, password, timeout):
    session = requests.Session()
    session.mount("https://", SSLAdapter())

    is_proxy = "8080/odata" in odata_url or "loca.lt/odata" in odata_url
    if is_proxy:
        params = {"username": username, "password": password, "$format": "json"}
        resp = session.get(
            odata_url, params=params, headers={"Accept": "application/json"},
            timeout=timeout, verify=False
        )
    else:
        auth = (username, password) if username and password else None
        resp = session.get(
            odata_url, auth=auth, headers={"Accept": "application/atom+xml"},
            timeout=timeout, verify=False
        )

    if resp.status_code != 200:
        raise Exception(f"OData fetch failed: {resp.status_code}\n{resp.text}")

    if "json" in resp.headers.get("Content-Type", ""):
        df = pd.DataFrame(resp.json().get("d", {}).get("results", []))
    else:
        df = parse_odata_xml(resp.text)

    return df

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🤖 SAP Odata ChatBot")

with st.sidebar:
    st.header("🧠 Gemini Setup")
    gemini_url = st.text_input(
        "REST Endpoint URL",
        value="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
    )
    gemini_key = st.text_input("Gemini API Key (AIza...)", type="password")
    timeout = st.number_input("Timeout (sec)", value=30, min_value=5, max_value=120)

    st.header("🌐 OData Configuration")
    odata_url = st.text_input("OData Service URL (EntitySet)", placeholder="https://server/sap/opu/odata/.../EntitySet")
    username = st.text_input("Username (optional)")
    password = st.text_input("Password (optional)", type="password")

    st.markdown("---")
    if st.button("🔄 Refresh OData Cache"):
        st.cache_data.clear()
        st.success("✅ OData cache cleared. Data will refetch on next run.")

if not gemini_url or not gemini_key:
    st.warning("Please enter your Gemini URL and API key.")
    st.stop()

if not odata_url:
    st.info("Enter OData service URL to fetch data.")
    st.stop()

# -----------------------------
# Fetch OData Data (cached)
# -----------------------------
st.write("📡 Fetching from:", odata_url)
try:
    df = fetch_odata_data(odata_url, username, password, timeout)
except Exception as e:
    st.error(f"❌ Failed to fetch or parse OData: {e}")
    st.stop()

# -----------------------------
# Prepare DataFrame
# -----------------------------
orig_cols = df.columns.tolist()
norm_map = {c: normalize_col(c) for c in orig_cols}
df.columns = [norm_map[c] for c in orig_cols]
reverse_map = {v: k for k, v in norm_map.items()}
fuzzy_map = fuzzy_column_map(df.columns)

for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors='ignore')

st.success(f"✅ Loaded {len(df)} rows from OData service (cached).")
st.dataframe(df.head(100))

# -----------------------------
# Gemini Prompt Template
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
6. Never use print() or display()
7. Always RETURN the final result (DataFrame, Series, numeric, dict, or plt.Figure)
8. Always handle NaN values safely:
   - For string filters: use str.contains(..., na=False)
   - For numeric operations: safely handle empty sequences
9. Always assume the expression will be executed inside a safe environment that automatically displays the result.
10. Do not print anything — simply return the result or expression output.
11. Prefer one-liners that evaluate to a result directly (no variables unless necessary).
12. When multiple values are logically related (like total count + list), return a dictionary.
13. If visualization is the best answer, generate a matplotlib figure object (plt.figure()) and plot accordingly.
14. When grouping numeric columns, use aggregation (sum, mean, count).
15. Do not answer general knowledge questions (outside dataset); reply with "only ask questions related to data please".
"""

# -----------------------------
# Ask Question
# -----------------------------
user_q = st.text_input("Ask your question about this OData data:")
if not user_q:
    st.stop()

with st.spinner("💡 Thinking with Gemini..."):
    resp = call_gemini_json(
        gemini_url,
        gemini_key,
        PROMPT_PANDAS_TRANSLATE + "\nQuestion: " + user_q,
        timeout
    )
    js = extract_json_from_response(resp)

if not js or "expr" not in js:
    msg = ""
    try:
        msg = resp["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        pass

    if msg:
        if "only ask questions related to data" in msg.lower():
            st.warning("💡 Gemini says: Only ask questions related to your OData data.")
        else:
            st.warning(f"🤖 Gemini replied:\n\n{msg}")
    else:
        st.error("❌ Gemini response parsing failed:")
        st.json(resp)
    st.stop()

expr = js["expr"]
explain = js.get("explain", "")

# -----------------------------
# Execute and Display
# -----------------------------
result = safe_exec(expr, df)

if isinstance(result, pd.DataFrame):
    st.markdown("### 📈 Result Table")
    st.dataframe(result)
elif isinstance(result, pd.Series):
    st.markdown("### 📊 Result Series")
    st.dataframe(result.to_frame())
elif isinstance(result, plt.Figure):
    st.markdown("### 📊 Visualization")
    st.pyplot(result)
else:
    fig = plt.gcf()
    if fig.get_axes():
        st.markdown("### 📊 Visualization")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.markdown(f"### ✅ Result: **{result}**")

# -----------------------------
# Natural Language Answer
# -----------------------------
PROMPT_ENGLISH = f"""
You are a helpful assistant. 
Question: {user_q}
The result is: {repr(result)}
Give the answer with explanation in simple English not like you are analyzing the data, explain to the user,just simple explanation is enougfh.
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
