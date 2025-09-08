
import os
from pydantic import BaseModel, Field
from langsmith import Client
from langchain_google_genai import ChatGoogleGenerativeAI
from rag_langgraph import runnable


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=os.environ["GOOGLE_API_KEY"],  
)

DATASET_NAME = "RAG Pilot v1"
client = Client()

class CorrectnessGrade(BaseModel):
    explanation: str = Field(..., description="Explain your reasoning for the score")
    correct: bool = Field(..., description="True if the answer is correct, False otherwise.")

correctness_instructions = """You are a teacher grading a quiz.
You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER.
Grade ONLY factual accuracy vs the ground truth. Extra correct info is OK; no conflicts.
Return 'correct=True' only if it fully matches without contradictions.
Explain your reasoning first, then the label.
"""

correctness_llm = llm.with_structured_output(CorrectnessGrade)

def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    gold = reference_outputs.get("gold_answer", "")
    msg = (
        f"QUESTION: {inputs['question']}\n"
        f"GROUND TRUTH ANSWER: {gold}\n"
        f"STUDENT ANSWER: {outputs.get('answer','')}"
    )
    grade = correctness_llm.invoke([
        {"role": "system", "content": correctness_instructions},
        {"role": "user", "content": msg},
    ])
    return bool(grade.correct)

class RelevanceGrade(BaseModel):
    explanation: str = Field(..., description="Explain your reasoning for the score")
    relevant: bool = Field(..., description="Does the answer address the question?")

relevance_instructions = """You are a teacher grading a quiz.
You will be given a QUESTION and a STUDENT ANSWER.
Return relevant=True only if the answer addresses the question and helps the user.
Explain your reasoning first, then the label.
"""

relevance_llm = llm.with_structured_output(RelevanceGrade)

def relevance(inputs: dict, outputs: dict) -> bool:
    msg = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs.get('answer','')}"
    grade = relevance_llm.invoke([
        {"role": "system", "content": relevance_instructions},
        {"role": "user", "content": msg},
    ])
    return bool(grade.relevant)

class GroundedGrade(BaseModel):
    explanation: str = Field(..., description="Explain your reasoning for the score")
    grounded: bool = Field(..., description="True if fully supported by the facts")

grounded_instructions = """You are a teacher grading groundedness.
You will be given FACTS (retrieved context) and a STUDENT ANSWER.
Return grounded=True only if all claims are supported by the FACTS (no hallucinations).
Explain your reasoning first, then the label.
"""

grounded_llm = llm.with_structured_output(GroundedGrade)

def groundedness(inputs: dict, outputs: dict) -> bool:
    docs = outputs.get("documents") or []
    doc_string = "\n\n".join(d.get("page_content","") for d in docs)
    msg = f"FACTS:\n{doc_string}\n\nSTUDENT ANSWER:\n{outputs.get('answer','')}"
    grade = grounded_llm.invoke([
        {"role": "system", "content": grounded_instructions},
        {"role": "user", "content": msg},
    ])
    return bool(grade.grounded)


class RetrievalRelevanceGrade(BaseModel):
    explanation: str = Field(..., description="Explain your reasoning for the score")
    relevant: bool = Field(..., description="True if retrieved docs are relevant to the question")

retrieval_relevance_instructions = """You are grading if the retrieved FACTS are relevant to the QUESTION.
If ANY portion is semantically related, consider them relevant. Minor off-topic content is OK.
Explain your reasoning first, then the label.
"""

retrieval_relevance_llm = llm.with_structured_output(RetrievalRelevanceGrade)

def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    docs = outputs.get("documents") or []
    doc_string = "\n\n".join(d.get("page_content","") for d in docs)
    msg = f"QUESTION:\n{inputs['question']}\n\nFACTS:\n{doc_string}"
    grade = retrieval_relevance_llm.invoke([
        {"role": "system", "content": retrieval_relevance_instructions},
        {"role": "user", "content": msg},
    ])
    return bool(grade.relevant)


def target(inputs: dict) -> dict:
    q = inputs["question"]
    result = runnable.invoke({"question": q})


    answer = result.get("answer", "")

   
    raw_docs = (result.get("documents") or result.get("contexts") or result.get("chunks") or [])
    if isinstance(raw_docs, str):
        raw_docs = [raw_docs]
    documents = [{"page_content": str(d)} for d in raw_docs]

    return {"answer": answer, "documents": documents}


dataset = client.read_dataset(dataset_name = DATASET_NAME)
examples = list(client.list_examples(dataset_id=dataset.id))

BATCH_SIZE = 3
for i in range (0, len(examples), BATCH_SIZE):
    batch = examples[i:1 + BATCH_SIZE]



experiment = client.evaluate(
    target,
    data=batch, 
    evaluators=[correctness, relevance, groundedness, retrieval_relevance],
    experiment_prefix="rag-batch",
    metadata={"grader_model": "gemini-1.5-flash"},
    max_concurrency=1,
)

print("Single-row eval complete.")
try:
    print(experiment.to_pandas())
except Exception:
    pass