# app.py

import os
import requests
import streamlit as st

# Environment-configurable API URL
API_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Financial 10-K RAG Assistant",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Financial 10-K RAG Assistant")
st.caption("Decoupled Architecture: Streamlit UI → FastAPI Backend → Ollama Engine")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar control panel
with st.sidebar:
    st.header("⚙️ System Status")
    st.info(f"Target API Engine:\n`{API_URL}`")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Display historical messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "route" in message and message["role"] == "assistant":
            st.caption(f"🔀 Route: `{message['route']}`")
        if "contexts" in message and message["contexts"]:
            with st.expander("📚 View Retrieved Sources"):
                for idx, ctx in enumerate(message["contexts"], 1):
                    st.markdown(f"**Source Chunk {idx}:**\n```\n{ctx}\n```")

# Handle user input
if prompt := st.chat_input("Ask a question about 3M, Adobe, or financial metrics..."):
    # Append user prompt to history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process response via backend API
    with st.chat_message("assistant"):
        with st.spinner("Processing query via FastAPI backend..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={"query": prompt},
                    timeout=180
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "")
                    route_name = data.get("route", "UNKNOWN")
                    contexts = data.get("contexts", [])

                    st.markdown(answer)
                    st.caption(f"🔀 Route Selected: `{route_name}`")

                    if contexts:
                        with st.expander("📚 View Retrieved Sources"):
                            for idx, ctx in enumerate(contexts, 1):
                                st.markdown(f"**Source Chunk {idx}:**\n```\n{ctx}\n```")

                    # Record assistant response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "route": route_name,
                        "contexts": contexts
                    })
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")

            except Exception as e:
                st.error(f"Failed to communicate with API backend at {API_URL}: {e}")