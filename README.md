# AI Research Paper Assistant

This repository contains a small Python application that loads a research paper (PDF), splits it into logical sections, builds a FAISS vector index over sectionised chunks, and exposes a Streamlit UI for searching and browsing the document with a Retrieval-Augmented Generation (RAG) pipeline.

## 📁 Project Structure

```
AI_ResearchPaper_Assistant/
├── Faiss_index.py        # builds FAISS index with section metadata
├── parser.py             # PDF loader and section splitter
├── rag_engine.py         # retrieval + LLM answer generator
├── main.py               # simple CLI wrapper (optional)
├── app.py                # Streamlit user interface
├── env/                  # Python virtual environment (not committed)
├── README.md             # this file
└── TrafficFlowGAN_.pdf   # example paper (removed by default)
```

## 🛠️ Requirements

- Python 3.10 (or later)
- `venv` or other virtual environment manager

### Python dependencies

All required packages are listed in `requirements.txt`. Install them via:

```bash
pip install -r requirements.txt
```

**Key packages:**
- `streamlit` – web UI framework
- `langchain`, `langchain-community`, `langchain-huggingface`, `langchain-openai` – LLM orchestration
- `faiss-cpu` – semantic search index
- `PyMuPDF` – PDF text extraction
- `sentence-transformers` – embeddings
- `python-dotenv` – environment variable management
- `pandas` – data manipulation

Using the included `env` directory is one fast way to ensure the correct environment; activate it with:

```bash
source env/bin/activate
```

## 🚀 Running the Application

1. **Activate the environment** (if not already active):
   ```bash
   source env/bin/activate
   ```

2. **Install dependencies** (if starting fresh without the pre-built env):
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Streamlit UI**:
   ```bash
   ./env/bin/streamlit run app.py
   ```

   or

   ```bash
   streamlit run app.py
   ```

   Make sure Streamlit is invoked from the same Python environment where the packages were installed.

4. **Upload a PDF research paper** using the sidebar uploader.
   - The app will split the document into sections, chunk each section, build/update the FAISS index, and display navigation options.
   - Enter natural language queries to ask the model about the paper. Responses are generated via the RAG pipeline.

5. **Browse sections** using the "Browse Sections" tab. Content is displayed in a scrollable text area.

6. **View paper overview** statistics in the "Paper Overview" tab.

## 🧩 Key Components

### `parser.py`
- Uses `PyMuPDFLoader` to extract text from each page of a PDF.
- Implements `split_into_sections()` to detect numbered and keyword headings and return `(heading, content)` pairs.

### `Faiss_index.py`
- Loads text via `parser.load_pdf` and `split_into_sections`.
- Splits sections into chunks using `RecursiveCharacterTextSplitter`.
- Builds an in-memory FAISS index using `HuggingFaceEmbeddings` with optional caching and section metadata.

### `rag_engine.py`
- Configures a retriever from the FAISS vector store and a cross-encoder reranker for better relevance.
- Constructs a prompt template and runs an LLM (`ChatOpenAI` with a Groq API endpoint in this case) to answer queries based on retrieved chunks.

### `app.py`
- Streamlit front end providing:
  - PDF upload capability
  - Dynamic navigation based on upload
  - Search interface with answer display and source chunk visibility
  - Section browsing and overview statistics
- Uses caching to avoid rebuilding the index unnecessarily and clears cache when a new file is uploaded.

### `main.py`
- A simple command-line wrapper for quick testing of `RAG_Engine` if a paper is already loaded.

## ⚠️ Notes & Tips

- **Caching behavior**: `get_vectorstore_and_sections` is cached with `@st.cache_resource`. When uploading a new file, the cache is explicitly cleared to ensure the index rebuilds.

- **Environment issues**: Always run Streamlit and Python commands from the same environment (`env` directory) to avoid `ModuleNotFoundError` errors.

- **Customization**: You can modify the section-heading regex in `parser.py` or adjust chunk size/overlap in `Faiss_index.py` to suit different document styles.

- **Dependencies**: For reproducibility, consider generating a `requirements.txt` via `pip freeze > requirements.txt`.

## 🧪 Testing the App

- Upload a known PDF such as `TrafficFlowGAN_.pdf` to verify sections are detected correctly.
- Ask questions like "What is TrafficFlowGAN?" to see RAG responses.

## 📦 Distribution

This project is designed for local use and exploration. If you wish to package it:

1. Dependencies are already listed in `requirements.txt` – simply run `pip install -r requirements.txt` for any new environment.
2. Optionally containerize with Docker for portability.
3. Provide instructions for setting up API keys (e.g., Groq or OpenAI) in `.env`.

---

Happy experimenting with your AI research paper assistant! Let me know if you'd like help adding more features or deploying the app.
