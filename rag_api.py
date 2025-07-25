from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel 
import psycopg2 
import os 
from dotenv import load_dotenv
import google.generativeai as genai


app = FastAPI()
load_dotenv()
genai.configure(api_key="AIzaSyDqU6ZEgbpyiRWV8QJuaGrujc61rmVs1Ag")

def get_connection():
    return psycopg2.connect(
        database ="postgres",
        user="postgres",
        password=os.getenv("DB_PASSWORD"),
        host="localhost",
        port="5432"
    )
class Query(BaseModel):
    question:str

@app.post("/ask")
def ask_question(query: Query):
    response = genai.embed_content(
        model="models/embedding-001", 
        content=query.question, 
        task_type="retrieval_query",
    )
    query_embedding = response["embedding"]
    query_embedding_str = "[" + ", ".join(map(str,query_embedding)) + "]"

    conn = get_connection()
    cur=conn.cursor()
    cur.execute(
        """
        SELECT text
        FROM documents 
        ORDER BY embedding <-> %s::vector
        LIMIT 3 
        """,
        (query_embedding_str,)
    )
    
    results = cur.fetchall()
    cur.close()
    conn.close()
    return {"top_chunks": [row[0] for row in results]}