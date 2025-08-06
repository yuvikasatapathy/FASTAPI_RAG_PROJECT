
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-pro")

def generate_answer_from_chunks(question, top_chunks):
    context = "\n\n".join(chunk[0] for chunk in top_chunks)
    prompt = f"Use the following information to answer the question:\n\n{top_chunks}\n\nQuestion: {question}"
    response = model.generate_content(prompt)
    return response.text

def stream_answer_from_chunks(question, top_chunks): 
    import time 
    context = "\n\n".join(chunk[0] for chunk in top_chunks)
    prompt = f"Use the following information to answer the question:\n\n{context}\n\nQuestion: {question}"
    response = model.generate_content(prompt)
    full_text = response.text 
    for sentence in full_text.split(" ."):
        yield sentence.strip() + "."
        time.sleep(0.3)