from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,END
from langgraph.prebuilt import ToolNode
from typing import TypedDict,Annotated
import operator

retrieval_context = {}


def set_retrieval_context(model,collection,bm25,chunks,reranker_r):
    retrieval_context["model"] = model
    retrieval_context["collection"] = collection
    retrieval_context["bm25"] = bm25
    retrieval_context["chunks"] = chunks
    retrieval_context["reranker_r"] = reranker_r

@tool
def search_context(query, model=None, collection=None, bm25=None, chunks=None, reranker_r=None):
    """Search the uploaded PDF documents for information relevant to the query.
     Use this when the question is about content that might be in the uploaded documents."""
    from rag_backend import hybrid_retriever, reranker

    model = retrieval_context.get("model", model)
    collection = retrieval_context.get("collection", collection)
    bm25 = retrieval_context.get("bm25", bm25)
    chunks = retrieval_context.get("chunks", chunks)
    reranker_r = retrieval_context.get("reranker_r", reranker_r)

    if collection is None or bm25 is None or chunks is None:
        return "No document collection is available yet. Please upload a PDF first."

    hybrid_results = hybrid_retriever(query, collection, chunks, bm25, top_k=8)
    final_results = reranker(query, hybrid_results, reranker_r, top_n=3)
    if not final_results:
        print("any document not found..")
        return "No relevant document content was found for this query."

    context = "\n\n".join(
        [f"[{r['metadata'].get('source_file', 'unknown')}] {r['document']}" for r in final_results]
    )
    return context

@tool
def calculator(expression:str)->str:
    """perform mathematical calculations.Use this for any math questions 
    like addition, multiplication, percentages, etc. Input should be a valid Python math expression like '25 * 4'"""

    try:
        result =eval(expression,{"__builtins__":{}},{})
        return f"the result is {result}"
    except Exception as e:
        return f"could not calculated: {str(e)}"

class AgentState(TypedDict):
    messages:Annotated[list,operator.add]

def create_agent_llm(groq_api_key):
    llm = ChatGroq(model = "openai/gpt-oss-120b",temperature=0.1,groq_api_key=groq_api_key)
    tools = [search_context,calculator]
    llm_with_tools = llm.bind_tools(tools)
    return llm_with_tools,tools

def agent_node(state:AgentState,llm_with_tools):
    """agent is thinking.."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages" : [response]}

def should_continue(state:AgentState):
    """check tool is required"""
    last_messages = state["messages"][-1]
    if last_messages.tool_calls:
        return "use_tool"
    else:
        return "end"

def build_agent_graph(groq_api_key):
    llm_with_tools,tools = create_agent_llm(groq_api_key)

    graph = StateGraph(AgentState)
    graph.add_node("agent",lambda state:agent_node(state,llm_with_tools))
    graph.add_node("tools",ToolNode(tools))
    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"use_tool":"tools","end":END}
    )
    graph.add_edge("tools","agent")
    return graph.compile()

def run_agent(query,groq_api_key):
    agent_executor =  build_agent_graph(groq_api_key)
    system_msg = {
            "role": "system", 
            "content": "You are a helpful assistant with access to tools. When using search_documents, base your answer strictly on the retrieved content and keep it concise. Do not add extensive outside knowledge "
            "or lengthy formatting unless the user explicitly asks for detail. pls fooloow this given terms"
        }
    result = agent_executor.invoke({"messages":[system_msg,{"role":"user","content":query}]})

    tools_used = []
    retrieved_text = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_used.append(tc["name"])
        if hasattr(msg, "name") and msg.name == "search_context":  # apna actual tool naam daal
            retrieved_text.append(msg.content)
    final_messages = result["messages"][-1]
    return final_messages.content,tools_used,retrieved_text

