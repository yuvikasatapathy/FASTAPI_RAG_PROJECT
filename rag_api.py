from fastapi import FastAPI, HTTPException, UploadFile, File 
from pydantic import BaseModel 
import psycopg2 
import os 
from dotenv import load_dotenv
import google.generativeai as genai
import shutil 
from rag_langgraph import runnable 
from langgraph_workflow import build_graph

app = FastAPI()
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

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
    graph = build_graph()
    result = graph.invoke({"question": query.question})
    return result 




@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        temp_file_path = f"./temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        result = runnable.invoke({"text": temp_file_path})
        return {
            "message": "PDF processed and stored in database successfully",
            "result": result
        }
    except Exception as e:
        # Print the error in the console and raise an HTTP 500 for better debugging
        print("❌ ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))