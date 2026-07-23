"""
Streamlit Web UI Dashboard.

Provides an interactive chat interface communicating with the FastAPI backend,
along with deep retrieval inspection sidebars to visualize RRF scores and Cross-Encoder ranks.
"""

import streamlit as st
import requests
import pandas as pd

# Configure Streamlit page
st.set_page_config(
    page_title="Invariable Docs",
    page_icon="📚",
    layout="wide",
)

API_URL = "http://localhost:8000/api/v1"

def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_retrieved_chunks" not in st.session_state:
        st.session_state.last_retrieved_chunks = []
    if "latency" not in st.session_state:
        st.session_state.latency = 0.0

init_session_state()

# -----------------------------------------------------------------------------
# Sidebar: Retrieval Inspection & Settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ RAG Settings")
    st.markdown("Configure hybrid retrieval pipelines dynamically.")
    
    use_hyde = st.toggle("Use HyDE (Query Expansion)", value=True)
    top_k = st.slider("Initial Dense/Sparse Top K", min_value=5, max_value=50, value=15, step=5)
    final_n = st.slider("Final Cross-Encoder Top N", min_value=1, max_value=10, value=4, step=1)
    
    st.divider()

    st.title("📄 Document Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Ingest Document"):
            with st.spinner("Ingesting document (Chunking & Embedding)..."):
                import os
                
                # Save uploaded file to disk so FastAPI can read it
                upload_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "uploads"))
                os.makedirs(upload_dir, exist_ok=True)
                file_path = os.path.join(upload_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Call FastAPI backend
                try:
                    payload = {"file_path": file_path}
                    resp = requests.post(f"{API_URL}/ingest", json=payload)
                    resp.raise_for_status()
                    result = resp.json()
                    st.success(f"Successfully ingested {result['chunks_processed']} chunks!")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    st.divider()
    
    st.title("🔍 Retrieval Inspection")
    if not st.session_state.last_retrieved_chunks:
        st.info("No chunks retrieved yet. Ask a question!")
    else:
        st.success(f"Pipeline latency: {st.session_state.latency}s")
        for idx, chunk in enumerate(st.session_state.last_retrieved_chunks):
            score = chunk.get("score", 0.0)
            doc_id = chunk.get("metadata", {}).get("doc_id", "Unknown")
            page_no = chunk.get("metadata", {}).get("page_no", "?")
            
            with st.expander(f"Rank {idx+1} | Score: {score:.3f} | [Doc: {doc_id}]"):
                st.markdown(f"**Page {page_no}**")
                st.caption(chunk.get("text", ""))


# -----------------------------------------------------------------------------
# Main Chat Interface
# -----------------------------------------------------------------------------
st.title("📚 Invariable Docs Knowledge Assistant")
st.markdown("Enterprise-grade Hybrid-Search RAG, running 100% locally on Apple Silicon.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI Backend
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳ *Thinking (Searching Qdrant & Generating with Ollama)...*")
        
        try:
            payload = {
                "query": prompt,
                "top_k": top_k,
                "final_top_n": final_n,
                "use_hyde": use_hyde,
                "filters": {}
            }
            response = requests.post(f"{API_URL}/query", json=payload)
            response.raise_for_status()
            
            data = response.json()
            answer = data.get("answer", "")
            
            # Update session state for sidebar inspection
            st.session_state.last_retrieved_chunks = data.get("retrieved_chunks", [])
            st.session_state.latency = data.get("latency_sec", 0.0)
            
            # Display answer
            message_placeholder.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            error_msg = f"❌ API Error: {str(e)}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
