import streamlit as st
import os
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder, SentenceTransformer
from rank_bm25 import BM25Okapi
import re
import tempfile

from rag_backend import load_pdf, fixed_size_chunking, genrate_embedding, create_vector_store, add_document_collection,check_faithfulness
from agent_backend import run_agent, set_retrieval_context

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Agent Demo", page_icon="🤖")
st.title("🤖 Agentic RAG Demo")
st.caption("This agent decides on its own whether to search your documents or use a calculator tool, based on your question.")

if not GROQ_API_KEY:
    st.error("API key not configured.")
    st.stop()


uploaded_files = st.file_uploader("Upload PDFs to enable document search for the agent", type="pdf", accept_multiple_files=True)

@st.cache_resource
def setup_agent_system(file_data_list):
    temp_dir = tempfile.mkdtemp()
    for filename, filebytes in file_data_list:
        with open(os.path.join(temp_dir, filename), "wb") as f:
            f.write(filebytes)
    
    docs = load_pdf(temp_dir)
    chunks = fixed_size_chunking(docs)
    
    bge_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    texts = [c.page_content for c in chunks]
    embeddings = genrate_embedding(texts, bge_model)
    
    collection = create_vector_store()
    add_document_collection(collection, chunks, embeddings)
    
    tokenized_corpus = [re.findall(r'\w+', t.lower()) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    return bge_model, collection, bm25, chunks, reranker
if uploaded_files:
    file_data_list = tuple((f.name, f.getvalue()) for f in uploaded_files)

    with st.spinner("Setting up agent's knowledge base..."):
        model, collection, bm25, chunks, reranker = setup_agent_system(file_data_list)

    set_retrieval_context(model, collection, bm25, chunks, reranker)
    st.session_state.agent_system_ready = True
    st.success("Agent is ready. Ask a document question or a math question below.")

if st.session_state.get("agent_system_ready", False):
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []
    
    for msg in st.session_state.agent_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    user_query = st.chat_input("Ask a document question, or try a math question like 'what is 847 * 23'")
    
    if user_query:
        st.session_state.agent_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)
        
        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking..."):
                answer, tools_used,retrieved_text = run_agent(user_query, GROQ_API_KEY)
                st.write(answer)
                if "search_context" in tools_used and retrieved_text:
                    # Retrieved text ko fake "context_chunks" format mein convert karo, function reuse karne ke liye
                    fake_chunks = [{"document": t} for t in retrieved_text]
                    faithfulness_data = check_faithfulness(answer, fake_chunks, model)
                    
                    if not faithfulness_data.get("is_refusal", False):
                        st.metric("Faithfulness (document-grounded)", f"{faithfulness_data['faithfulness_percent']}%")
                        with st.expander("Sentence-level breakdown"):
                            for d in faithfulness_data['details']:
                                icon = "✅" if d['grounded'] else "⚠️"
                                st.write(f"{icon} ({d['similarity']}) {d['sentence']}")
                if tools_used:
                    st.caption(f"Tool(s) used: {', '.join(set(tools_used))}")
                else:
                    st.caption("Answered directly, no tool needed")
        
        st.session_state.agent_messages.append({"role": "assistant", "content": answer})
else:
    st.info("Upload a PDF above to activate the agent's document search capability. You can still ask math questions without uploading anything.")