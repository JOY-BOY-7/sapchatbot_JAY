import streamlit as st
import pandas as pd
import requests

# ------------------------------
# CONFIGURATION
# ------------------------------
NLP_API_URL = "https://api.nlpcloud.io/v1/gpu/dolphin-mixtral-8x7b/chatbot"
NLP_API_KEY = "458273542354923845789462738452387423459"

headers = {
    "Authorization": f"Token {NLP_API_KEY}",
    "Content-Type": "application/json"
}

st.set_page_config(page_title="Excel Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Excel AI Chatbot (Dolphin Mixtral 8x7B)")
st.caption("Upload an Excel file and ask any question about its contents. Powered by NLP Cloud Dolphin Mixtral 8x7B.")

st.success("✅ Connected to NLP Cloud Dolphin Mixtral 8x7B Chatbot.")


# ------------------------------
# NLP CLOUD HELPER
# ------------------------------
def nlpcloud_chatbot(user_input, context="", history=[]):
    payload = {
        "input": user_input,
        "context": context,
        "history": history
    }
    resp = requests.post(NLP_API_URL, headers=headers, json=payload)
    if resp.status_code != 200:
        return f"⚠️ NLP Cloud Error {resp.status_code}: {resp.text}", history
    j = resp.json()
    response_text = j.get("response", "⚠️ No response in API output")
    new_history = j.get("history", history)
    return response_text, new_history


# ------------------------------
# EXCEL UPLOAD & PROCESSING
# ------------------------------
uploaded_file = st.file_uploader("📂 Upload an Excel file", type=["xlsx", "xls"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.session_state["excel_data"] = df
    st.success(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns from Excel.")
    st.dataframe(df.head(10))
else:
    st.info("Please upload an Excel file to continue.")


# ------------------------------
# CHATBOT INTERFACE
# ------------------------------
if "excel_data" in st.session_state:
    df = st.session_state["excel_data"]

    if "history" not in st.session_state:
        st.session_state["history"] = []

    st.subheader("💬 Ask a Question")

    question = st.text_input(
        "Enter your question:",
        placeholder="e.g. What is the total sales for 2024? or Who has the highest salary?"
    )

    if st.button("🤖 Ask AI"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            # Create a context summary of the Excel data
            context = f"""
You are an intelligent assistant. Use the following Excel data to answer accurately.
If the answer cannot be found, say so politely.

Here’s a snapshot of the data (first few rows):

{df.head(10).to_string(index=False)}
"""

            with st.spinner("Thinking..."):
                answer, new_history = nlpcloud_chatbot(
                    user_input=question,
                    context=context,
                    history=st.session_state["history"]
                )
                st.session_state["history"] = new_history

                st.markdown("### 🧠 Answer")
                st.write(answer)

else:
    st.stop()
