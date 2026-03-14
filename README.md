```markdown
# Cogniva — Universal Knowledge Assistant

**Cogniva** is an **Agentic Retrieval-Augmented Generation (RAG) chatbot with memory** that allows users to ask questions about **documents, YouTube videos, and web knowledge**.

Built with:  
**Python · Streamlit · Groq · FAISS · LangGraph · HuggingFace**

---

## Live App

Streamlit Deployment  
https://YOUR_STREAMLIT_LINK

---

## Quick Test (YouTube Example)

Paste this video URL into the app to test immediately:

https://www.youtube.com/watch?v=KSdPYtWlIMA

Example questions to try:

- Summarize the video  
- What is the main concept explained?  
- Explain the architecture mentioned in the video  

---

## Features

- Multi-document upload (PDF, DOCX, TXT, MD, CSV, JSON)
- YouTube transcript ingestion
- Retrieval-Augmented Generation using **FAISS**
- Agent-based reasoning with **LangGraph**
- Web search fallback (**Tavily / DuckDuckGo**)
- Source citations (file, page, timestamp)
- Conversational memory
- Concise / Detailed answer modes

---

## How It Works

1. Documents and transcripts are **chunked and converted to embeddings** using `all-MiniLM-L6-v2`.
2. Embeddings are stored in a **FAISS vector database**.
3. When a user asks a question:
   - Relevant chunks are retrieved via similarity search.
   - If no match is found → web search is used.
4. A **LangGraph ReAct agent** decides which tool to use.
5. The **Groq LLM** generates the final answer with source citations.

---

## Run Locally

Clone the repository

```

git clone [https://github.com/YOUR_USERNAME/cogniva.git](https://github.com/YOUR_USERNAME/cogniva.git)
cd cogniva

```

Create environment

```

python -m venv venv

```

Activate

Windows
```

venv\Scripts\activate

```

Mac/Linux
```

source venv/bin/activate

```

Install dependencies

```

pip install -r requirements.txt

```

Create `.env`

```

GROQ_API_KEY=your_api_key

```

Run the app

```

streamlit run app.py

```

---

## Tech Stack

- **LLM:** Groq (Llama 3)
- **Embeddings:** HuggingFace sentence-transformers
- **Vector DB:** FAISS
- **Agent Framework:** LangGraph
- **UI:** Streamlit
- **Web Search:** Tavily / DuckDuckGo

---

## Deployment

Streamlit Cloud  
https://YOUR_STREAMLIT_LINK

---

## Author

Sandra Praveen C
```
