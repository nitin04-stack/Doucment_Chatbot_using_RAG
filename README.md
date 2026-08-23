# RAG Chatbot with Hybrid Search
A multi-document question answering system that lets you upload PDFs and ask questions grounded strictly in their content. Built to explore and demonstrate a production-style RAG pipeline, not just a basic retrieval demo.

## What it does

You upload one or more PDFs, and the app builds a searchable knowledge base from them. You can then ask questions in a chat interface, and the system retrieves the most relevant passages and generates an answer using only that retrieved context, with sources cited.

## Key features

**Hybrid retrieval**
Combines semantic search (embedding similarity via BGE) with keyword search (BM25) instead of relying on embeddings alone. This matters because pure semantic search can miss exact terms, numbers, or specific phrases that a keyword-based approach catches. The two scores are normalized and combined with a weighted average.

**Re-ranking**
After hybrid retrieval narrows down candidates, a cross-encoder model re-scores the top results by processing the query and each document together, giving a more accurate final ranking than either search method alone.

**Multi-document support with citations**
Multiple PDFs can be uploaded in a single session. Every retrieved chunk carries its source filename and page number, so answers can be traced back to the exact document they came from.

**Session-based chat history**
Each set of uploaded PDFs starts a new chat session, tracked with a name, timestamp, and its own conversation history. Past sessions are listed in the sidebar and can be reopened or deleted individually.

**Automatic document summaries**
After processing, each uploaded PDF gets a short auto-generated title, topic, and page count, so you know what you are working with before asking anything.

**Backend-managed API access**
The Groq API key is configured on the backend (via environment variables or Streamlit secrets) rather than asked from the user, so the app works out of the box without requiring anyone to bring their own key.

## How it works

1. PDFs are loaded and split into overlapping chunks (500 characters, 100 character overlap).
2. Each chunk is embedded using the BGE-small sentence-transformer model and stored in a ChromaDB vector store, alongside a BM25 keyword index built from the same chunks.
3. On a query, hybrid retrieval pulls candidates using both semantic similarity and keyword overlap, and combines the two scores.
4. The top candidates are re-ranked using a cross-encoder for more accurate relevance ordering.
5. The final passages are passed to a Groq-hosted LLM with a strict prompt that restricts answers to the provided context.

## Tech stack

- **Frontend/App**: Streamlit
- **Embeddings**: sentence-transformers (BAAI/bge-small-en-v1.5)
- **Vector store**: ChromaDB
- **Keyword search**: BM25 (rank_bm25)
- **Re-ranking**: cross-encoder (ms-marco-MiniLM-L-6-v2)
- **LLM**: Groq (openai/gpt-oss-120b)
- **PDF parsing**: LangChain's PyPDFLoader

## Project structure

```
rag-chatbot/
├── app.py              # Streamlit UI, session management, chat logic
├── rag_backend.py       # Core RAG pipeline: chunking, embeddings, retrieval, generation
├── requirements.txt     # Dependencies
└── .gitignore            # Excludes API keys and local environment files
```

## Running it locally

1. Clone the repository and move into the project folder.
2. Create a virtual environment and activate it.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the project root with your Groq API key:
   ```
   GROQ_API_KEY=your_key_here
   ```
5. Run the app:
   ```
   streamlit run app.py
   ```

## Notes and limitations

- The vector store is in-memory by default, so it rebuilds each time a new session starts. This keeps the setup simple but means there is no persistence across app restarts.
- The re-ranking step adds latency compared to using hybrid search alone, but improves the accuracy of the top results, which matters more for answer quality than raw speed here.
- Answers are constrained to the uploaded documents by design. If a question falls outside what the documents cover, the system is prompted to say so rather than guess.

## Author
Nitin Kumawat
