
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-pro")

def generate_answer_from_chunks(question, top_chunks):
    context = "\n\n".join(chunk[0] for chunk in top_chunks)
    prompt = f"Use the following information to answer the question:\n\n{top_chunks}\n\nQuestion: {question}"
    response = model.generate_content(prompt)
    return response.text
