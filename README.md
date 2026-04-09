# Ask-From-Web

A lightweight RAG app built as a trial project for Inferyx. Point it at any website, and ask questions about its content. No manual reading, no copy-pasting — just a URL and a question.

---

## Origin

Built as an exploratory trial to test whether web-scraped content could be queried conversationally using a RAG pipeline. The notebook (`notebook.ipynb`) was the initial proof-of-concept — a single URL in, a question in, an answer out. `app.py` is the full Streamlit version built from that prototype, with persistent FAISS storage, multi-URL support, and source attribution.

---

## Notebook → App: What Changed

| | `notebook.ipynb` | `app.py` |
|--|-----------------|---------|
| Input | Single URL via `input()` | `url.txt` — multiple URLs loaded on startup |
| Storage | In-memory FAISS, lost on restart | Persisted to `dat_embeding.pkl` on disk |
| Deduplication | None | Skips URLs already in the index |
| Answer format | Raw LLM output | Structured — Overview, Explanation, References |
| Source tracking | None | URL number mapping saved to `url_mapping.json` |
| Interface | Terminal | Streamlit UI with stats |

The notebook confirmed the core idea works. The app productionised it.

---

## How it Works

```
url.txt
    ↓
WebBaseLoader scrapes each URL
    ↓
RecursiveCharacterTextSplitter
chunk_size=1000, chunk_overlap=500
    ↓
HuggingFace all-MiniLM-L6-v2 embeddings
    ↓
FAISS vector store — persisted to disk
    ↓
User asks question → retriever fetches top 4 chunks
    ↓
Gemini 1.5 Flash generates structured answer
with Overview / Explanation / References
```

---

## Files

| File | What it is |
|------|-----------|
| `app.py` | Full Streamlit app — scrape, embed, query |
| `notebook.ipynb` | Original proof-of-concept (single URL, terminal) |
| `url.txt` | Input — one URL per line |
| `index.faiss` + `index.pkl` | Pre-built FAISS index (Inferyx docs) |
| `inferyx_docs.json` | Scraped Inferyx documentation content |
| `inferyx_doc_links.json` | URL-to-ID mapping for Inferyx docs |
| `data.json` | Scraped content from current URL set |
| `requirements.txt` | Dependencies |

---

## Setup

```bash
git clone https://github.com/parthrudrawar/Ask-From-Web.git
cd Ask-From-Web

pip install -r requirements.txt

# Add your API key
echo "GOOGLE_API_KEY=your_key_here" > .env

# Add URLs to scrape
echo "https://example.com" >> url.txt

streamlit run app.py
```

On startup, the app automatically reads `url.txt`, scrapes all URLs, embeds the content, and loads the FAISS index. If an index already exists on disk it loads that instead of re-scraping.

---

## Example Questions

```
What is a Datapod in Inferyx?
How does the pipeline execution work?
What are the steps to configure a datasource?
```

---

## Tech Stack

`Streamlit` · `LangChain` · `Gemini 1.5 Flash` · `FAISS` · `all-MiniLM-L6-v2` · `BeautifulSoup` · `WebBaseLoader`

---

## Author

[parthrudrawar](https://github.com/parthrudrawar)
