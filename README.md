# 🔬 Multi-Document Research Assistant

> A RAG (Retrieval-Augmented Generation) chatbot that lets users upload
> multiple documents (PDF, DOCX, TXT) and get AI-powered, source-cited
> answers — built entirely with free and open-source tools.

🔗 **Live Demo**: [Add your Streamlit Cloud link here]

---

## ✨ Features

| Feature | Details |
|---|---|
| 📄 Multi-format upload | PDF, DOCX, TXT — multiple files at once |
| 🧠 Local embeddings | `sentence-transformers/all-MiniLM-L6-v2` — runs on CPU, zero cost |
| 🗄️ Persistent vector store | FAISS — fast, file-based, no server required |
| 🤖 Dual LLM modes | 🌐 Online (Groq LLaMA 3.3 70B) or 💻 Offline (Ollama, fully local) |
| 📎 Source citations | Every answer shows which document it came from |
| 📋 Document manifest | Always reports the correct total document count, even for meta-questions |
| 💬 Chat history | Full conversation with export to `.txt` |
| 🧹 Session management | Clear chat / clear all documents (with safe Windows file handling) |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/multi-document-rag-assistant.git
cd multi-document-rag-assistant
```

### 2. Create an environment (Conda recommended)

```bash
conda create -n rag_assistant python=3.12 -y
conda activate rag_assistant
```

> Using `venv` instead works too:
> ```bash
> python -m venv venv
> # Windows: venv\Scripts\activate
> # macOS/Linux: source venv/bin/activate
> ```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> ⚠️ First run downloads the embedding model (~80 MB) from HuggingFace. This is a one-time step — after that, embeddings run fully offline.

### 4. Get a FREE Groq API key (for Online mode)
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free) → API Keys → Create Key
3. Copy the key (starts with `gsk_...`)

### 5. (Optional) Save key in `.env`
```bash
cp .env.example .env
# Edit .env and paste your key
```
Or just paste it directly into the app's sidebar at runtime — both work.

### 6. Run the app
```bash
streamlit run app.py
```
Open your browser at **http://localhost:8501**

---

## 🌐 Online vs 💻 Offline Mode

The app supports two LLM backends, switchable from the sidebar:

| Mode | Engine | Internet Required? | Setup |
|---|---|---|---|
| 🌐 **Online** | Groq — LLaMA 3.3 70B | Yes | Paste free API key in sidebar |
| 💻 **Offline** | Ollama — LLaMA 3.2 (local) | No | One-time setup below |

### Offline mode setup (optional)
```bash
# 1. Install Ollama from https://ollama.com

# 2. Pull a model (one-time, ~2GB)
ollama pull llama3.2

# 3. Install the connector
pip install langchain-ollama
```
Embeddings and the vector store already run 100% locally in both modes — only the final answer-generation step differs.

---

## 🗂️ Project Structure

```
multi-document-rag-assistant/
│
├── app.py              ← Streamlit UI (sidebar, chat, mode toggle)
├── rag_pipeline.py     ← Core RAG logic (parse, chunk, embed, retrieve, answer)
│
├── requirements.txt    ← Python dependencies
├── .env.example        ← Copy to .env and add your API key
├── .gitignore          ← Keeps secrets and large files out of git
│
└── faiss_store/        ← Auto-created: vector index + manifest (gitignored)
```

---

## 🏗️ How It Works

```
User uploads PDF/DOCX/TXT
        │
        ▼
  Parse raw text
  (PyMuPDF / python-docx)
        │
        ▼
  Split into overlapping chunks
  (RecursiveCharacterTextSplitter)
        │
        ▼
  Embed each chunk locally
  (sentence-transformers MiniLM, cached singleton)
        │
        ▼
  Store in FAISS index (persists to disk)
  + update document manifest (manifest.json)
        │
        ▼
  User asks a question
        │
        ▼
  Embed the question (same cached model)
        │
        ▼
  Retrieve top-8 relevant chunks (similarity search)
        │
        ▼
  Build prompt: retrieved chunks + full document
  manifest (for accurate "how many docs" answers)
        │
        ▼
  Send prompt → Groq (online) or Ollama (offline)
        │
        ▼
  Return cited answer to user
```

---

## ⚙️ Configuration

Edit the constants at the top of `rag_pipeline.py`:

| Variable | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 800 | Characters per chunk |
| `CHUNK_OVERLAP` | 150 | Overlap between chunks |
| `TOP_K` | 8 | Chunks retrieved per query |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Online LLM model |
| `OLLAMA_MODEL` | `llama3.2` | Offline local LLM model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model |

---

## 🚢 Deploy to Streamlit Community Cloud (Free)

1. Push this repo to GitHub (`faiss_store/` and `.env` are excluded automatically)
2. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub
3. Click **New app** → select this repo → main file: `app.py`
4. (Optional) Add `GROQ_API_KEY` under **Advanced settings → Secrets**
5. Deploy — you'll get a public `*.streamlit.app` URL

---

## 🛠️ Tech Stack

- **[LangChain](https://langchain.com)** — RAG orchestration
- **[FAISS](https://github.com/facebookresearch/faiss)** — Local vector search
- **[sentence-transformers](https://sbert.net)** — Free local embeddings
- **[Groq](https://groq.com)** — Free LLaMA 3.3 70B API (online mode)
- **[Ollama](https://ollama.com)** — Local LLM runtime (offline mode)
- **[Streamlit](https://streamlit.io)** — Web UI
- **[PyMuPDF](https://pymupdf.readthedocs.io)** & **python-docx** — Document parsing

---

## 👤 Author

**Muhammad Adnan**
Data Science Student | AI & NLP Enthusiast | Building RAG & Chatbot Applications

- 💻GitHub: [(https://github.com/Eden3311)]
- LinkedIn: [www.linkedin.com/in/muhammad-adnan-9732612b4]
- 📧 Email: [edenrose372@gmail.com]


## 📄 License
MIT — free to use, modify, and share.
