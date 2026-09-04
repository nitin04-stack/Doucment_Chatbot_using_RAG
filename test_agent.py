from agent_backend import set_retrieval_context,run_agent
from rag_backend import build_rag_system
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import os

load_dotenv()
groq_api_key = os.getenv("groq_api_key")

print("system is loading...")
model,collection,bm25,chunks = build_rag_system("data/pdfs")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

set_retrieval_context(model,collection,bm25,chunks,reranker)
print("Ready..")

answer1 = run_agent("what is multi-head attention ",groq_api_key)
print(answer1)

answer2 = run_agent("what is 847 multipliedd by 23",groq_api_key)
print(answer2)