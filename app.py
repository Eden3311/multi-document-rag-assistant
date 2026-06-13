"""
app.py
------
Streamlit front-end for the Multi-Document Research Assistant.

Pages / sections:
  • Sidebar  — API key input, document upload & management
  • Main     — Chat interface with cited answers
  • Footer   — Session stats

Run with:
    streamlit run app.py
"""

import os
import time
import tempfile
from pathlib import Path

import streamlit as st

# Import our RAG pipeline
from rag_pipeline import (
    ingest_file,
    answer_question,
    load_vector_store,
    clear_vector_store,
)


# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be the very first Streamlit call)
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS  — clean dark-accent professional theme
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Global typography ───────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Header banner ───────────────────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    border-left: 5px solid #3b82f6;
}
.app-header h1 {
    color: #f8fafc;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.app-header p {
    color: #94a3b8;
    margin: 0;
    font-size: 0.95rem;
}

/* ── Chat messages ───────────────────────────────────────────────── */
.user-message {
    background: #1e40af;
    color: #eff6ff;
    border-radius: 18px 18px 4px 18px;
    padding: 14px 18px;
    margin: 8px 0 8px 60px;
    font-size: 0.95rem;
    line-height: 1.6;
    box-shadow: 0 2px 8px rgba(30,64,175,0.3);
}
.assistant-message {
    background: #1e293b;
    color: #e2e8f0;
    border-radius: 18px 18px 18px 4px;
    padding: 16px 20px;
    margin: 8px 60px 8px 0;
    font-size: 0.95rem;
    line-height: 1.7;
    border: 1px solid #334155;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
.assistant-message code {
    font-family: 'JetBrains Mono', monospace;
    background: #0f172a;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.85rem;
    color: #7dd3fc;
}

/* ── Source citation cards ───────────────────────────────────────── */
.source-card {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.82rem;
    color: #94a3b8;
}
.source-card strong {
    color: #60a5fa;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Stat pills (sidebar) ────────────────────────────────────────── */
.stat-pill {
    display: inline-block;
    background: #1e3a5f;
    color: #7dd3fc;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 3px 2px;
}

/* ── File chip (uploaded file indicator) ────────────────────────── */
.file-chip {
    background: #064e3b;
    color: #6ee7b7;
    border: 1px solid #065f46;
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 0.8rem;
    margin: 4px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Section labels ──────────────────────────────────────────────── */
.section-label {
    color: #64748b;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin: 16px 0 8px 0;
}

/* ── Input box ───────────────────────────────────────────────────── */
.stTextInput > div > div > input {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #f1f5f9 !important;
    border-radius: 10px !important;
}

/* ── Buttons ─────────────────────────────────────────────────────── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

/* ── Scrollable chat area ────────────────────────────────────────── */
.chat-container {
    max-height: 520px;
    overflow-y: auto;
    padding-right: 8px;
}

/* ── Empty state ─────────────────────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: #475569;
}
.empty-state .icon { font-size: 3rem; margin-bottom: 16px; }
.empty-state h3 { color: #64748b; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ════════════════════════════════════════════════════════════════════════════

def init_session():
    """Initialise all session-state keys once per browser session."""
    defaults = {
        "chat_history":      [],        # list of {"role": "user"|"assistant", "content": ..., "sources": [...]}
        "indexed_files":     [],        # list of filenames successfully indexed
        "total_chunks":      0,         # cumulative chunks across all uploads
        "groq_api_key":      "",        # user-supplied key
        "api_key_verified":  False,     # True once user confirms the key works
        "llm_mode":          "online",  # "online" (Groq) or "offline" (Ollama)
        "uploader_key":      0,         # incremented to force file_uploader reset
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:

    # ── App logo / name ──────────────────────────────────────────────────
    st.markdown("## 🔬 ResearchRAG")
    st.markdown('<div class="section-label">Mode</div>', unsafe_allow_html=True)

    # ── Online / Offline mode toggle ─────────────────────────────────────
    mode_choice = st.radio(
        label="LLM Mode",
        options=["🌐 Online (Groq — fast, needs internet)", "💻 Offline (Ollama — local, no internet)"],
        index=0 if st.session_state.llm_mode == "online" else 1,
        label_visibility="collapsed",
    )
    st.session_state.llm_mode = "online" if mode_choice.startswith("🌐") else "offline"

    if st.session_state.llm_mode == "offline":
        st.markdown(
            '<span class="stat-pill">💻 Offline mode — Ollama (llama3.2)</span>',
            unsafe_allow_html=True,
        )
        with st.expander("⚙️ Offline setup (one-time)"):
            st.markdown("""
1. Install **Ollama** from [ollama.com](https://ollama.com)
2. Open terminal and run:
   ```
   ollama pull llama3.2
   ```
3. Make sure Ollama is running (it starts automatically after install)
4. Install the connector:
   ```
   pip install langchain-ollama
   ```

✅ No API key needed — embeddings + LLM both run on your machine.
            """)

    st.markdown('<div class="section-label">Configuration</div>', unsafe_allow_html=True)

    # ── Groq API key input (only needed for online mode) ────────────────
    if st.session_state.llm_mode == "online":
        groq_key_input = st.text_input(
            "Groq API Key",
            type="password",
            value=st.session_state.groq_api_key,
            placeholder="gsk_...",
            help="Free key from groq.com → Console → API Keys",
        )

        if groq_key_input:
            st.session_state.groq_api_key = groq_key_input
            st.markdown(
                '<span class="stat-pill">✅ Key saved</span>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown('<div class="section-label">Upload Documents</div>', unsafe_allow_html=True)

    # ── File uploader ────────────────────────────────────────────────────
    # The 'key' includes a counter that we increment when "Clear All
    # Documents" is clicked. This forces Streamlit to render a BRAND NEW
    # empty uploader widget, discarding any previously selected files —
    # otherwise the same files would be detected as "new" again and
    # re-indexed automatically after clearing.
    uploaded_files = st.file_uploader(
        label="Drag & drop or browse",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Supported: PDF, DOCX, TXT",
        label_visibility="collapsed",
        key=f"file_uploader_{st.session_state.uploader_key}",
    )

    # ── Process uploaded files ───────────────────────────────────────────
    if uploaded_files:
        new_files = [
            f for f in uploaded_files
            if f.name not in st.session_state.indexed_files
        ]

        if new_files:
            with st.spinner(f"Indexing {len(new_files)} file(s)…"):
                for uf in new_files:
                    try:
                        # Save to a temp file so our parser can read it
                        suffix = Path(uf.name).suffix
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=suffix
                        ) as tmp:
                            tmp.write(uf.read())
                            tmp_path = tmp.name

                        num_chunks, src_name = ingest_file(tmp_path, original_name=uf.name)
                        os.unlink(tmp_path)   # clean up temp file

                        st.session_state.indexed_files.append(uf.name)
                        st.session_state.total_chunks += num_chunks

                        st.success(f"✅ {src_name} — {num_chunks} chunks indexed")

                    except Exception as e:
                        st.error(f"❌ {uf.name}: {str(e)}")

    # ── Show indexed files ───────────────────────────────────────────────
    if st.session_state.indexed_files:
        st.markdown('<div class="section-label">Indexed Documents</div>', unsafe_allow_html=True)

        for fname in st.session_state.indexed_files:
            st.markdown(
                f'<div class="file-chip">📄 {fname}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", len(st.session_state.indexed_files))
        with col2:
            st.metric("Chunks", st.session_state.total_chunks)

        # Clear everything button
        if st.button("🗑️ Clear All Documents", use_container_width=True):
            clear_vector_store()
            st.session_state.indexed_files  = []
            st.session_state.total_chunks   = 0
            st.session_state.chat_history   = []
            st.session_state.uploader_key  += 1   # forces a fresh, empty uploader
            st.rerun()

    st.markdown("---")

    # ── Help / how-to ────────────────────────────────────────────────────
    with st.expander("ℹ️ How to use"):
        st.markdown("""
1. **Get a free Groq API key** at [groq.com](https://console.groq.com)
2. **Paste the key** in the field above
3. **Upload** one or more PDF / DOCX / TXT files
4. **Ask questions** in the chat — answers cite sources
5. Upload more files anytime to expand the knowledge base
        """)


# ════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ════════════════════════════════════════════════════════════════════════════

# ── Header banner ────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🔬 Multi-Document Research Assistant</h1>
    <p>Upload your documents and ask anything — powered by LLaMA 3.3 70B + local embeddings</p>
    <p style="margin-top: 8px; font-size: 0.8rem; color: #64748b;">
        Developed by: <strong>Muhammad Adnan Contact: +923272999009</strong>
    </p>
</div>
""", unsafe_allow_html=True)


# ── Chat history ──────────────────────────────────────────────────────────────
chat_placeholder = st.container()

with chat_placeholder:
    if not st.session_state.chat_history:
        # Empty state
        st.markdown("""
        <div class="empty-state">
            <div class="icon">💬</div>
            <h3>No conversation yet</h3>
            <p>Upload a document in the sidebar, then ask your first question below.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:

            if msg["role"] == "user":
                st.markdown(
                    f'<div class="user-message">🧑 {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

            else:
                # Assistant answer
                st.markdown(
                    f'<div class="assistant-message">🤖 {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

                # Source citations
                if msg.get("sources"):
                    st.markdown(
                        '<div class="section-label" style="margin-top:8px;">📎 Sources</div>',
                        unsafe_allow_html=True,
                    )
                    for src in msg["sources"]:
                        st.markdown(
                            f"""<div class="source-card">
                                <strong>📄 {src['source']}</strong><br>
                                {src['snippet']}…
                            </div>""",
                            unsafe_allow_html=True,
                        )

                st.markdown("<br>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# QUESTION INPUT  (always visible at the bottom)
# ════════════════════════════════════════════════════════════════════════════

st.markdown("---")

# We use a form so pressing Enter submits the question
with st.form(key="question_form", clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1])

    with col_input:
        user_question = st.text_input(
            label="Ask a question",
            placeholder="e.g. What are the main findings in the report?",
            label_visibility="collapsed",
        )

    with col_btn:
        submit = st.form_submit_button("Send ➤", use_container_width=True)


# ── Handle submission ────────────────────────────────────────────────────────
if submit and user_question.strip():

    # Guard: API key required only for online mode
    if st.session_state.llm_mode == "online" and not st.session_state.groq_api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar first.")
        st.stop()

    # Guard: documents required
    if not st.session_state.indexed_files:
        st.warning("⚠️ Please upload at least one document before asking questions.")
        st.stop()

    # Add user message to history
    st.session_state.chat_history.append({
        "role":    "user",
        "content": user_question,
    })

    # ── Run the RAG pipeline ─────────────────────────────────────────────
    spinner_text = (
        "🔍 Searching documents and generating answer…"
        if st.session_state.llm_mode == "online"
        else "🔍 Searching documents and generating answer locally (Ollama)…"
    )
    with st.spinner(spinner_text):
        try:
            answer, sources = answer_question(
                question=user_question,
                mode=st.session_state.llm_mode,
                groq_api_key=st.session_state.groq_api_key,
            )
        except Exception as e:
            answer  = f"❌ Error: {str(e)}\n\nPlease check your settings and try again."
            sources = []

    # Add assistant message to history
    st.session_state.chat_history.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources,
    })

    # Rerun to refresh the chat display
    st.rerun()


# ── Extra actions row ────────────────────────────────────────────────────────
if st.session_state.chat_history:
    col_a, col_b, _ = st.columns([1, 1, 4])
    with col_a:
        if st.button("🔄 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    with col_b:
        # Download entire conversation as a text file
        convo_text = "\n\n".join(
            f"[{m['role'].upper()}]\n{m['content']}"
            for m in st.session_state.chat_history
        )
        st.download_button(
            label="⬇️ Export Chat",
            data=convo_text,
            file_name="research_conversation.txt",
            mime="text/plain",
        )
