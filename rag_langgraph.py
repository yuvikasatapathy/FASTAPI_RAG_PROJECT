from langgraph.graph import StateGraph, END 
from typing import TypedDict, Annotated, List, Dict, Any, Literal, Optional 
from langchain_core.runnables import Runnable
import os 
from dotenv import load_dotenv
import psycopg2
from PyPDF2 import PdfReader
import nltk 
from nltk.tokenize import PunktSentenceTokenizer
import google.generativeai as genai 
from psycopg2 import OperationalError

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
nltk.download("punkt", quiet=True)
class RAGState(TypedDict, total= False):
    text:str
    chunks: List[str]
    embeddings: List[List[float]]
    top_results: list[str]
    question: str 
    answer: str 
    documents: List[Dict[str, Any]]
def extract_text_from_pdf(state: RAGState) -> RAGState:
    pdf_path = state.get("text")
    if not pdf_path:
        return {"text":""} 
    try: 
        reader = PdfReader(pdf_path)
        raw = "".join(page.extract_text() or "" for page in reader.pages)
        return {"text": raw}
    except Exception: 
        return {"text": pdf_path}
def chunk_text(state: RAGState) -> RAGState:
    raw = state.get("text") or ""
    if not raw.strip():
        return {"chunks": []}
    pst = PunktSentenceTokenizer()
    sentences = pst.tokenize(raw)
    return {"chunks": sentences}
def embed_chunks(state: RAGState) -> RAGState:
    embedded =[]
    chunks = state.get("chunks") or []
    for ch in chunks:
        res = genai.embed_content(
            model="models/embedding-001",
            content=ch,
            task_type="retrieval_document"
        )
        embedded.append(res["embedding"])
    return {"embeddings": embedded}
def store_to_db(state:RAGState) -> RAGState:
    chunks = state.get("chunks") or []
    embs = state.get("embeddings") or []
    if not chunks or not embs: 
        return {}
    conn = psycopg2.connect(
      database="postgres",
        user="postgres",
        password=os.getenv("DB_PASSWORD"),
        host="127.0.0.1",
        port="5432"  
    )
    try: 
        cur = conn.cursor()
        for text, emb in zip(chunks, embs):
            cur.execute("INSERT INTO documents (text, embedding) VALUES (%s, %s::vector)", (text, emb))
        conn.commit()
        cur.close()
    finally: 
        conn.close()
    return{}
def retrieve(state: RAGState) -> RAGState:
    """Given a question, pull top-k similar texts from pgvector.
       Safe: never crashes eval; returns empty docs on failure."""
    q = (state.get("question") or "").strip()
    if not q:
        return {"top_results": [], "documents": []}

    try:
        q_emb = genai.embed_content(
            model="models/embedding-001",
            content=q,
            task_type="retrieval_query",
        )["embedding"]
    except Exception as e:
        return {"top_results": [], "documents": [], "embed_error": str(e)}

    if os.getenv("SKIP_DB", "0") == "1":
        return {"top_results": [], "documents": []}

    conn = None
    cur = None
    try:
        conn = psycopg2.connect(
            database="postgres",
            user="postgres",
            password=os.getenv("DB_PASSWORD"),
            host="127.0.0.1",
            port="5432",
        )
        cur = conn.cursor()
        cur.execute(
            """
            SELECT text
            FROM documents
            ORDER BY embedding <#> %s::vector  -- cosine distance (pgvector)
            LIMIT %s
            """,
            (q_emb, 5),
        )
        rows = cur.fetchall() or []
        docs = [r[0] for r in rows]
        return {"top_results": docs, "documents": [{"page_content": t} for t in docs]}
    except OperationalError as e:
        return {"top_results": [], "documents": [], "db_error": str(e)}
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


  
def answer_with_gemini(state: RAGState) -> RAGState:
    q = state.get("question") or ""
    docs = state.get("documents") or []
    ctx = "\n\n".join(d.get("page_content", "") for d in docs)

    prompt = (
        "You are a helpful assistant answering questions strictly from the provided context.\n"
        "If the answer is not in the context, say you don't know.\n\n"
        f"CONTEXT:\n{ctx}\n\nQUESTION: {q}\n\nANSWER:"
    )
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(prompt)
    ans = (getattr(resp, "text", "") or "").strip()
    return {"answer": ans}

def route(state: RAGState) -> Literal["index", "qa"]:
    """
    Decide which flow to run:
    - 'index' if we were given a PDF path or raw text in `text`
    - 'qa'    if we were given a `question`
    Default to 'qa' so evals don't crash when no `text` is present.
    """
    if (state.get("text") or "").strip():
        return "index"
    return "qa"

graph = StateGraph(RAGState)
graph.add_node("extract_text", extract_text_from_pdf)
graph.add_node("chunk", chunk_text)
graph.add_node("embed", embed_chunks)
graph.add_node("store", store_to_db)

graph.add_node("retrieve", retrieve)
graph.add_node("answer", answer_with_gemini)

graph.set_entry_point("extract_text") 
from langgraph.graph import START
def choose_after_extract(state: RAGState) -> str:
    return "chunk" if route(state) == "index" else "retrieve"

graph.add_conditional_edges(
    "extract_text",
    choose_after_extract,
    {"chunk": "chunk", "retrieve": "retrieve"},
)

graph.add_edge("chunk", "embed")
graph.add_edge("embed", "store")
graph.add_edge("store", END)

graph.add_edge("retrieve", "answer")
graph.add_edge("answer", END)

runnable = graph.compile()