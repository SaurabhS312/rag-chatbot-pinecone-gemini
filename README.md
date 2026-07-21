# RAG Chatbot — Chat with Your PDFs

<img width="1355" height="524" alt="image" src="https://github.com/user-attachments/assets/52947745-4b15-4a00-aed3-58f862991d33" />

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload a PDF and ask questions about its contents. Built as a hands-on project while learning GenAI/RAG engineering.

## How it works

1. **Upload** a PDF via the Streamlit sidebar
2. **Chunk** — the document is extracted (`pdfplumber`) and split into overlapping chunks (`RecursiveCharacterTextSplitter`)
3. **Embed** — each chunk is converted into a vector using a local HuggingFace embedding model (`BAAI/bge-small-en-v1.5`)
4. **Store** — vectors are upserted into a **Pinecone** serverless vector index
5. **Retrieve** — when you ask a question, the most relevant chunks are pulled back using MMR (Maximal Marginal Relevance) search
6. **Generate** — the retrieved context is passed to **Google Gemini** (via LangChain) to produce a grounded, context-only answer

```
PDF → Chunk → Embed (HuggingFace) → Store (Pinecone) → Retrieve (MMR) → Answer (Gemini)
```

## Tech stack

| Layer | Tool |
|---|---|
| UI | [Streamlit](https://streamlit.io) |
| Orchestration | [LangChain](https://www.langchain.com) (LCEL chains) |
| PDF parsing | [pdfplumber](https://github.com/jsvine/pdfplumber) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | HuggingFace `BAAI/bge-small-en-v1.5` (local, free) |
| Vector database | [Pinecone](https://www.pinecone.io) (serverless, free tier) |
| LLM | Google Gemini (`gemini-2.5-flash`) |
| Automation | GitHub Actions (scheduled Pinecone cleanup) |

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/SaurabhS312/rag-chatbot-pinecone-gemini.git
cd rag-chatbot-pinecone-gemini
```

**2. Create a virtual environment and install dependencies**
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**3. Add your API keys**

Create a `.env` file in the project root:
```
PINECONE_API_KEY=your_pinecone_key_here
GEMINI_API_KEY=your_gemini_key_here
```
- Get a free Pinecone key at [app.pinecone.io](https://app.pinecone.io)
- Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com)

**4. Run the app**
```bash
streamlit run ragChatBotLive.py
```

## Live demo

🔗 *[Add your deployed Streamlit Community Cloud link here once deployed]*

## Automated maintenance

A scheduled GitHub Actions workflow (`.github/workflows/cleanup.yml`) wipes the Pinecone index every ~15 days, keeping usage within the free tier as multiple users test the app. The index is automatically recreated the next time a file is uploaded — no manual intervention needed.

## What I learned building this

- Working with embeddings (word, sentence, and image) and vector similarity search
- Building and querying a vector database (Pinecone) end-to-end
- Chunking strategies for real-world documents
- Comparing retrieval approaches (top-k vs. MMR)
- Building LLM-orchestrated pipelines with LangChain's LCEL syntax
- Handling multi-user isolation in a shared vector store (Pinecone namespaces)
- Managing secrets safely (`.env`, `.gitignore`, GitHub Actions secrets)
- Deploying and maintaining a live GenAI app on a free-tier budget

## Roadmap

- [ ] Namespace isolation per uploaded file (avoid cross-document contamination)
- [ ] Support multi-file / multi-document Q&A
- [ ] Add source citations (page numbers) to answers
- [ ] Swap in agentic RAG (LangGraph) for multi-step reasoning

---

Built by [Saurabh](https://github.com/SaurabhS312) as part of a self-directed transition into GenAI/AI Data Engineering roles.
