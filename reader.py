from pypdf import PdfReader 
import nltk 
nltk.download("punkt")
from nltk.tokenize import PunktSentenceTokenizer
import os 
from dotenv import load_dotenv 
import google.generativeai as genai 
import psycopg2

reader = PdfReader("/Users/richa/Downloads/sample_medical_insurance.pdf")
number_of_pages = len(reader.pages)
page = reader.pages[0]
text = page.extract_text() 

pst = PunktSentenceTokenizer()
sentences = pst.tokenize(text)



load_dotenv()
genai.configure(api_key = os.getenv("GEMINI_API_KEY"))
data_chunks= [] 
for sentence in sentences: 
    response = genai.embed_content(
        model="models/embedding-001", 
        content=sentence, 
        task_type="retrieval_document"
    )
    embedding = response["embedding"]
    data_chunks.append({
        "text": sentence, 
        "embedding": embedding
    })
    for i, chunk in enumerate(data_chunks[:5]):
        print(f"[{i+1}] {chunk['text'][:60]}... → embedding length: {len(chunk['embedding'])}")


def store_embeddings(data_chunks):
    conn=psycopg2.connect(
        database="postgres",
        user="postgres",
        password = "DB_PASSWORD",
        host="localhost",
        port="5432"
    )
    cur=conn.cursor()
    for chunk in data_chunks:
        cur.execute(
                "INSERT INTO documents (text,embedding) VALUES( %s, %s::vector)",
                (chunk["text"], chunk["embedding"])
        )
    conn.commit()
    cur.close()
    conn.close()

store_embeddings(data_chunks)

def search_similar_chunks(query_embedding, top_k=3):
    conn = psycopg2.connect(
        database="postgres",
        user="postgres",
        password=os.getenv("DB_PASSWORD"),
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()
    cur.execute(
        """
        SELECT text
        FROM documents
        ORDER BY embedding <-> %s::vector
        LIMIT %s
        """,
        (query_embedding, top_k)
    )
    results = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    return results

result = genai.embed_content(
    model = "models/embedding-001",
    content = [
        "What type of surgeries does my insurance cover?",
        "Can I perform a cosmetic surgery?",
        "When does my insurance expire?",
        "What is my policy number?"
    ],
    task_type = "retrieval_query",
)
def embed_query(query):
    response = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )
    return response["embedding"]

print(result['embedding'])
def search_pgvector(query_embedding, top_k=3):
    conn=psycopg2.connect(
        database="postgres",
        user="postgres",
        password = "DB_PASSWORD",
        host="localhost",
        port="5432"
    )
    cur=conn.cursor()
    cur.execute(
        """
        SELECT text
        FROM documents
        ORDER BY embedding <-> %s::vector
        LIMIT %s
        """,
        (query_embedding, top_k * 2)
    )
    results = cur.fetchall()
    cur.close()
    conn.close()
    seen = set()
    unique_chunks = []
    for row in results: 
        text = row[0].strip()
        if text not in seen:
            seen.add(text)
            unique_chunks.append(text)
        if len(unique_chunks) == top_k:
            break 
    return results
query_embedding = result['embedding'][0] 
results = search_pgvector(query_embedding)#select one query embedding and store it in variable, will use this vector in sql query 
print(results)