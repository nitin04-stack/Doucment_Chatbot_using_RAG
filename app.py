import streamlit as st
from rag_backend import (
    load_pdf,
    fixed_size_chunking,
    hybrid_retriever,
    reranker,
    genrate_answer,
    generate_pdf_summary,
    get_pdf_stats,
    check_faithfulness,
    check_answer_relevancy,
    check_context_precision


)
from sentence_transformers import CrossEncoder
import tempfile
import os
from datetime import datetime
import uuid
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")
try:
    groq_api_key = groq_api_key or st.secrets.get("groq_api_key", "")
except Exception:
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")

st.set_page_config(page_title = "RAG Doucment Chatbot",page_icon="📚",layout = "wide")

if not groq_api_key:
    st.error("API is not configure.come after sometime")
    st.stop()
st.title(" Dcoument Q&A Assistent")

if "all_sessions" not in st.session_state:
    st.session_state.all_sessions = {}  

if "current_session_state_id" not in st.session_state:
    st.session_state.current_session_state_id = None

st.sidebar.divider()
if st.sidebar.button("➕ New Chat (Upload PDF)",use_container_width=True):
    st.session_state.current_session_state_id = None
    st.rerun()

st.sidebar.divider()

uploaded_files = st.sidebar.file_uploader(
    "upload your pdfs",
    type="pdf",
    accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload atleast one file to get started")
    st.stop()

st.sidebar.divider()
st.sidebar.subheader("📂 Chat History")
if not st.session_state.all_sessions:
    st.sidebar.caption("Nothing in History")
else:
    sorted_sessions = sorted(
        st.session_state.all_sessions.items(),
        key = lambda x:x[1]["timestamp"],
        reverse=True
    )

    for sid ,session_data in sorted_sessions:
        col1,col2 = st.sidebar.columns([4,1])
        with col1:
            label = f"{session_data['name']}\n{session_data['timestamp']}"
            if st.button(label,key = f"select_{sid}",use_container_width=True):
                    st.session_state.current_session_state_id = sid
                    st.rerun()
        with col2:
            if st.button("🗑️",key = f"delete_{sid}"):
                del st.session_state.all_sessions[sid]
                if st.session_state.current_session_state_id == sid:
                    st.session_state.current_session_state_id = None
                st.rerun()
st.sidebar.divider()

if uploaded_files and st.session_state.current_session_state_id is None:
    with st.spinner("Processing PDFs..."):
        temp_dir = tempfile.mkdtemp()
        for file in uploaded_files:
            with open(os.path.join(temp_dir,file.name),"wb") as f:
                f.write(file.getbuffer())
        docs = load_pdf(temp_dir)
        chunks = fixed_size_chunking(docs)

        from sentence_transformers import SentenceTransformer
        from rag_backend import genrate_embedding,create_vector_store,add_document_collection
        from rank_bm25 import BM25Okapi
        import re
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        texts = [c.page_content for c in chunks]
        embeddings = genrate_embedding(texts,model)
        collection = create_vector_store()

        add_document_collection(collection,chunks,embeddings)
        tokenize_corpus = [re.findall(r"\w+",t.lower()) for t in texts]
        bm25 = BM25Okapi(tokenize_corpus)
        reranker_r = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        pdf_summary = {}
        for file in uploaded_files:
            pdf_summary[file.name] = generate_pdf_summary(chunks,groq_api_key,file.name)
            pdf_stats = [get_pdf_stats(docs,f.name) for f in uploaded_files]

        new_sid = str(uuid.uuid4())
        st.session_state.all_sessions[new_sid] = {
            "name":list(pdf_summary.values())[0]["title"][:40],
            "timestamp" : datetime.now().strftime("%d-%b-%y %H:%M"),
            "messages":[],
            "pdf_summary":pdf_summary,
            "pdf_stats":pdf_stats,
            "file_names":[f.name for f in uploaded_files],
            "model":model,"collection":collection,"bm25":bm25,
            "chunks":chunks,"reranker":reranker_r
        }
        st.session_state.current_session_state_id = new_sid
        st.rerun()
elif st.session_state.current_session_state_id is not None:
    sid = st.session_state.current_session_state_id
    session = st.session_state.all_sessions[sid]

    with st.container(border = True):

        st.subheader("📄 Uploaded Documents")
        cols = st.columns(len(session["pdf_stats"]))
        for i, stat in enumerate(session["pdf_stats"]):
            filename = stat["filename"]
            summary = session["pdf_summary"].get(filename,{"title":filename,"topic":"Unkown"})
            with cols[i]:
                with st.container(border=True):
                    st.markdown(f"**{summary['title']}**")
                    st.caption(f"Topic: {summary['topic']}")
                    st.caption(f"{stat['total_pages']} pages")
    st.divider()

    for msg in session["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if user_query := st.chat_input("Ask question about the document..."):
        session["messages"].append({"role":"user","content":user_query})
        with st.chat_message("user"):
            st.write(user_query)
        with st.chat_message("assistant"):
            with st.spinner("Processing....."):
                hybrid_results = hybrid_retriever(
                    user_query, session["collection"], session["chunks"],
                    session["bm25"], top_k=8
                )
                final_results = reranker(user_query, hybrid_results, session["reranker"])
                answer = genrate_answer(user_query,final_results,groq_api_key)
                st.write(answer)

                faithfulness_data = check_faithfulness(answer, final_results, session["model"])
                relevancy_data = check_answer_relevancy(user_query, answer, session["model"])
                precision_data = check_context_precision(user_query, final_results, session["model"])

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Faithfulness", f"{faithfulness_data['faithfulness_percent']}%")
                with col2:
                    st.metric("Answer Relevancy", f"{relevancy_data['relevancy_percent']}%")
                with col3:
                    st.metric("Context Precision", f"{precision_data['precision_percent']}%")
                st.caption(faithfulness_data['verdict'])

                with st.expander("metrices"):
                    for d in faithfulness_data['details']:
                        icon = "✅" if d['grounded'] else "⚠️"
                        st.write(f"{icon} ({d['similarity']}) {d['sentence']}")
                
                with st.expander("Sources"):
                    score = [r["rerank_score"] for r in final_results]
                    if final_results:
                        score = [r["rerank_score"] for r in final_results]
                        st.caption(f"Top score: {max(score):.4f}")
        session["messages"].append({"role": "assistant", "content": answer,"faithfulness": faithfulness_data, "relevancy": relevancy_data, "precision": precision_data})
        st.caption("AI can make mistakes. Please double-check answers",text_alignment="center")
    else:
        st.info("👈 upload PDF to start new chat..")

st.sidebar.caption("Connect with me..")
st.sidebar.caption("LinkedIn: [www.linkedin.com/in/nitin-ku04]")

st.markdown(
    """
    <style>
    .footer-credit {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        text-align: center;
        padding: 5px;
        font-size: 12px;
        color: gray;
        background-color: white;
        z-index: 999;
    }
    </style>
    <div class="footer-credit">Made by Nitin</div>
  
    """,
    unsafe_allow_html=True
)


