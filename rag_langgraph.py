from langgraph.graph import StateGraph, END 
from typing import TypedDict, Annotated 
from langchain_core.runnables import Runnable
import os 
from dotenv import load_dotenv
import psycopg2
from PyPDF2 import PdfReader
import nltk 
from nltk.tokenize import PunktSentenceTokenizer
import google.generativeai as genai 

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
nltk.download("punkt")
class RAGState(TypedDict):
    text:str
    chunks: list[str]
    embeddings: list[list[float]]
    top_results: list[str]
def extract_text_from_pdf(state: RAGState) -> RAGState:
    reader = PdfReader(state["text"])
    text = reader.pages[0].extract_text()
    return {"text": text}
def chunk_text(state: RAGState) -> RAGState:
    pst = PunktSentenceTokenizer()
    sentences = pst.tokenize(state["text"])
    return {"chunks": sentences}
def embed_chunks(state: RAGState) -> RAGState:
    embedded_chunks=[]
    for chunk in state["chunks"]:
        res = genai.embed_content(
            model="models/embedding-001",
            content=chunk,
            task_type="retrieval_document"
        )
        embedded_chunks.append(res["embedding"])
    return {"embeddings": embedded_chunks}
def store_to_db(state:RAGState) -> RAGState:
    conn = psycopg2.connect(
      database="postgres",
        user="postgres",
        password=os.getenv("DB_PASSWORD"),
        host="localhost",
        port="5432"  
    )
    cur = conn.cursor()
    for text, emb in zip(state["chunks"], state["embeddings"]):
        cur.execute("INSERT INTO documents (text, embedding) VALUES (%s, %s::vector)", (text, emb))
    conn.commit()
    cur.close()
    conn.close()
    return{}
graph = StateGraph(RAGState)
graph.add_node("extract_text", extract_text_from_pdf)
graph.add_node("chunk", chunk_text)
graph.add_node("embed", embed_chunks)
graph.add_node("store", store_to_db)
graph.set_entry_point("extract_text")
graph.add_edge("extract_text", "chunk")
graph.add_edge("chunk", "embed")
graph.add_edge("embed", "store")
graph.set_finish_point("store")
runnable = graph.compile()