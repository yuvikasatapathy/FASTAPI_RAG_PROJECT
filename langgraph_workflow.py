from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from reader import embed_query
from reader import search_similar_chunks
from llm import stream_answer_from_chunks


def retrieve_node(state):
    query = state["question"]
    query_embedding = embed_query(query)
    top_chunks = search_similar_chunks(query_embedding, top_k=3)
    return {"question": query, "chunks": top_chunks}


def prompt_node(state):
   
    chunks = state["chunks"]
    question = state["question"]
    context = "\n\n".join(chunks)
    prompt = f"Answer the following question based on the provided insurance document context.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    return {"prompt": prompt}


def llm_node(state):
    response = generate_answer_from_chunks(state["prompt"])
    return {"response": response}

def get_query_embedding_node(state):
    print ("[DEBUG] Node: get_query_embedding_node started")
    from reader import embed_query
    question = state["question"]
    embedding = embed_query(question)
    print(f"[DEBUG] Node: get_query_embedding_node finished")
    return{"query_embedding": embedding, 
           "question": question
    }
    
def search_pgvector_node(state):
    from reader import search_pgvector
    print("[DEBUG] Node: search_pgvector_node started")
    
    query_embedding = state["query_embedding"]
    question = state["question"]
    top_chunks = search_pgvector(query_embedding)
   
    print(f"[DEBUG] Node: search_pgvector_node finished")
    return {
        "top_chunks": top_chunks, 
        "question": question
    }

def call_llm_node(state):
    print("[DEBUG] Node: call_llm_node started")
 
    question = state["question"]
    chunks = state["top_chunks"]
    streamed_response =""
    for chunk in stream_answer_from_chunks(question,chunks):
        streamed_response += chunk + " "
    return {"answer": streamed_response.strip(), "top_chunks": chunks}
    print(f"[DEBUG] Node: call_llm_node finished")
def build_graph():

    workflow = StateGraph(dict)
    workflow.add_node("embed", get_query_embedding_node)
    workflow.add_node("search", search_pgvector_node)
    workflow.add_node("llm", call_llm_node)

    workflow.set_entry_point("embed")
    workflow.add_edge("embed", "search")
    workflow.add_edge("search", "llm")
    workflow.set_finish_point("llm")

    return workflow.compile()
