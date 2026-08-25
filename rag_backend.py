import os
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_groq import ChatGroq

def load_pdf(folder_path):
    all_doc = []
    num_pdf = 0

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path,filename)
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            for page in documents:
                page.metadata = {**page.metadata ,"source_file": filename}
            all_doc.extend(documents)
            num_pdf+=1
    print(f"number of pdf {num_pdf}")
    print(f"total number of documents {len(all_doc)}")
    return all_doc

def fixed_size_chunking(documents,chunk_size = 400,chunk_overlap = 40):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
    )
    return splitter.split_documents(documents)

em_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
def genrate_embedding(text,model = em_model):
    embeddings = model.encode(text)
    return embeddings

def create_vector_store():
    client = chromadb.PersistentClient(path = "./vector_store/chroma_db")
    collection = client.get_or_create_collection("streamlit_rag")
    return collection

def add_document_collection(collection,documents,embeddings):
    ids = []
    all_metadata = []
    document_content = []
    embeddings_list = []


    if len(documents) != len(embeddings):
        raise ValueError("documents and embeddings did not match")

    for i ,(doc,embedding) in enumerate(zip(documents,embeddings)):
        idd = f"id = {uuid.uuid4()}"
        ids.append(idd)

        metadata = dict(doc.metadata)
        metadata["doc_index"] = i
        metadata["content_length"] = len(doc.page_content)
        all_metadata.append(metadata)

        document_content.append(doc.page_content)

        embeddings_list.append(embedding.tolist())

    collection.add(
        ids=ids,
        metadatas=all_metadata,
        documents=document_content,
        embeddings=embeddings_list
    )

    print("Total document in vector store", len(document_content))
    print("document in collection ",collection.count())

def sementic_retriever(query,collection,top_k=10,score_threshold = 0.6):
    query_embeddings = genrate_embedding([query])[0]
    results = collection.query(
        query_embeddings = [query_embeddings.tolist()],
        n_results = top_k
    )

    retrieved_doc = []
    if results["documents"] and results["documents"][0]:
        ids = results["ids"][0]
        metadatas= results["metadatas"][0]
        documents = results["documents"][0]
        distances = results["distances"][0]

        for i ,(id,metadata,document,distance) in enumerate(zip(ids,metadatas,documents,distances)):
            similarity_score = 1 - distance

            if similarity_score >= score_threshold:
                retrieved_doc.append({
                    "ids":id,
                    "metadatas":metadata,
                    "documents":document,
                    "distances":distance,
                    "similarity_score":similarity_score,
                    "rank":i+1
                })
        print(f"total {len(retrieved_doc)} documents")
    else:
        print("no document found")
    return retrieved_doc


def build_rag_system(pdf_folder):
    docs = load_pdf(pdf_folder)
    chunk_doc = fixed_size_chunking(docs)
    texts = [chunk.page_content for chunk in chunk_doc]

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embeddings = genrate_embedding(texts)

    collection = create_vector_store()
    add_document_collection(collection,chunk_doc,embeddings)

    tokenized_corpus = [re.findall(r'\w+', t.lower()) for t in texts]
    if not tokenized_corpus or len(tokenized_corpus) == 0:
        raise ValueError("No text could be extracted from the uploaded PDF. Please check if the PDF contains selectable text or try another file.")
    bm25 = BM25Okapi(tokenized_corpus)

    return model, collection, bm25, chunk_doc

     
def hybrid_retriever(query, collection, chunks, bm25_index, top_k=10, alpha=0.6):
    sementic_results = sementic_retriever(query, collection, top_k=top_k)
    sementic_score = {r["documents"] : r["similarity_score"] for r in sementic_results}
    tokenize_query = re.findall(r"\w+",query.lower())
    bm25_score_raw = bm25_index.get_scores(tokenize_query)
    max_bm25 = max(bm25_score_raw) if max(bm25_score_raw) > 0 else 1
    normalized_bm25_score = {chunks[i].page_content: score/max_bm25 for i ,score in enumerate(bm25_score_raw)}

    combined_score = []

    for chunk in chunks:
        text = chunk.page_content
        sem_score = sementic_score.get(text, 0)
        kw_score = normalized_bm25_score.get(text, 0)
        final_score = alpha * sem_score + (1-alpha) * kw_score
        combined_score.append({
            "document":text,
            "metadata": chunk.metadata,
            "sementic_score":round(sem_score,4),
            "kw_score":round(kw_score,4),
            "final_score":round(final_score,4)
        })
    combined_score.sort(key = lambda x:x["final_score"],reverse=True)
    return combined_score[:top_k]

def reranker(query, hybrid_results, model=None, top_n=5):
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank = [[query,i["document"]] for i in hybrid_results]
    rerank_results = model.predict(rerank)

    for i ,score in enumerate(rerank_results):
        hybrid_results[i]["rerank_score"] = round(float(score),4)
    reranked = sorted(
        hybrid_results,
        key = lambda x:x["rerank_score"],
        reverse = True
    )
    return reranked[:top_n]

def genrate_answer(query,chunks,groq_api_key):
    groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        groq_api_key = groq_api_key,
        model = "openai/gpt-oss-120b",
        temperature = 0.1,
        max_tokens=1024
    )
    context = "\n\n".join([c["document"]for c in chunks])
    prompt = f"""Answer using ONLY the context below. If the context does not
contain enough information, say: "I don't have enough information to answer this."
Answer the query directly and concisely based only on the provided context without adding unnecessary fluff

CONTEXT:
{context}

QUESTION:
{query}

ANSWER (based strictly on the reference material):"""
    response = llm.invoke(prompt)
    return response.content


def generate_pdf_summary(chunks,groq_api_key,filename):
    groq_api_key = groq_api_key or os.getenv("groq_api_key")

    relevant_chunks = [c for c in chunks if c.metadata.get("source_file") == filename]
    if not relevant_chunks:
        return {"title": filename, "topic": "Unknown"}
    # sample_chunks = chunks[:4]
    sample_text = "\n\n".join([c.page_content for c in relevant_chunks[:5]])
    llm = ChatGroq(model = "openai/gpt-oss-120b",temperature=0.1,groq_api_key=groq_api_key)
    prompt = f"""Based on this excerpt from a document, provide:
    1. A likely title (one line)
    2. The main topic/subject (one line, e.g. "Machine Learning - Neural Networks")
    Answer the query directly and concisely based only on the provided context without adding unnecessary fluff
    excerpt: {sample_text} 
    Respond in this exact format:
    TITLE: <title>
    TOPIC: <topic>  """

    response = llm.invoke(prompt)
    result_text = response.content

    title = "UNKNOWN"
    topic = "General"

    for line in result_text.split("\n"):
        if line.startswith("TITLE:"):
            title = line.replace("TITLE:","").strip()
        elif line.startswith("TOPIC:"):
            topic = line.replace("TOPIC:","").strip()
    return{"title":title,"topic":topic}

def get_pdf_stats(docs,filename):
        page_of_file = [d for d in docs if d.metadata.get("source_file")==filename]
        return {"filename":filename,"total_pages":len(page_of_file)}

def split_into_Text(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip())>10]
    return sentences

def check_faithfulness(answer,context_chunks,model,threshold = 0.6):
    context_texts = [c["document"]for c in context_chunks]
    context_embeddings = model.encode(context_texts)

    answer_sentences = split_into_Text(answer)
    if not answer_sentences:
        return {
            "faithfullness_percent":100,
            "hallucination_percent":0,
            "details":[],
            "verdict":"No content to check"
        }
    grounded_count = 0
    details = []
    for sentence in answer_sentences:
        sentence_embedding = model.encode([sentence])
        best_similarity = 0

        for ctx_emb in context_embeddings:
            similarity = np.dot(sentence_embedding,ctx_emb)/(
                 np.linalg.norm(sentence_embedding) * np.linalg.norm(ctx_emb)
            )
            best_similarity = max(best_similarity,similarity)
        is_grounded = best_similarity>=threshold
        if  is_grounded:
            grounded_count += 1
        details.append({
            "sentence": sentence, 
            "similarity": round(float(np.asarray(best_similarity).item()), 3),
            "grounded": is_grounded
        })

    faithfulness_percent = round((grounded_count / len(answer_sentences)) * 100, 1)
    hallucination_percent = round(100 - faithfulness_percent, 1)

    if faithfulness_percent >= 80:
        verdict = "Low hallucination risk"
    elif faithfulness_percent >= 50:
        verdict = "Moderate hallucination risk"
    else:
        verdict = "High hallucination risk"
    return {
        "faithfulness_percent": faithfulness_percent,
        "hallucination_percent": hallucination_percent,
        "details": details,
        "verdict": verdict
    }

def check_answer_relevancy(query, answer, model):
    query_embedding = model.encode([query])[0]
    answer_embedding = model.encode([answer])[0]
    similarity = np.dot(query_embedding, answer_embedding) / (
        np.linalg.norm(query_embedding) * np.linalg.norm(answer_embedding)
    )
    return {"relevancy_percent": round(float(similarity) * 100, 1)}


def check_context_precision(query, context_chunks, model, threshold=0.65):
    if not context_chunks:
        return {"precision_percent": 0}
    
    query_embedding = model.encode([query])[0]
    relevant_count = 0
    for chunk in context_chunks:
        chunk_embedding = model.encode([chunk["document"]])[0]
        similarity = np.dot(query_embedding, chunk_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
        )
        if similarity >= threshold:
            relevant_count += 1
    
    return {"precision_percent": round((relevant_count / len(context_chunks)) * 100, 1)}


