# streamlit_app.py — The visual web interface
# Streamlit turns Python code into a web UI automatically!

import streamlit as st
import requests

API_URL = "http://localhost:8000"  # FastAPI backend address

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("📄 PDF Q&A Chatbot")
st.caption("Upload any PDF and ask questions about it")

# ── SIDEBAR ──
with st.sidebar:
    st.header("Upload PDF")
    uploaded = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded and st.button("Process PDF"):
        with st.spinner("Processing... (may take 30-60s)"):
            res = requests.post(
                f"{API_URL}/upload",
                files={"file": (uploaded.name, uploaded, "application/pdf")},
            )
        if res.status_code == 200:
            st.success("✅ PDF ready! Ask questions below.")
        else:
            st.error(res.json().get("detail"))

# ── CHAT INTERFACE ──
# st.session_state persists data across reruns (like a page refresh)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input box at the bottom
if question := st.chat_input("Ask a question about your PDF..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            res = requests.post(f"{API_URL}/ask", json={"question": question})
        if res.status_code == 200:
            data = res.json()
            st.write(data["answer"])
            if data["sources"]:
                with st.expander("📎 View sources"):
                    for s in data["sources"]:
                        st.caption(f"Page {s['page']}: {s['snippet']}")
            st.session_state.messages.append(
                {"role": "assistant", "content": data["answer"]}
            )
