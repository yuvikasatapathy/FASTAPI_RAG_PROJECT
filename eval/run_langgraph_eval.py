
from typing import List
from typing_extensions import Annotated, TypedDict
from langsmith import Client
from langchain_openai import ChatOpenAI


from rag_langgraph import runnable  


DATASET_NAME = "RAG Pilot v1"   
GRADER_MODEL = "gpt-4o"         

client = Client()
llm = ChatOpenAI(model=GRADER_MODEL, temperature=0)

# CORRECTNESS: Response vs Reference Answer

class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]

correctness_instructions = """You are a teacher grading a quiz.
You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER.
Grade ONLY factual accuracy vs the ground truth. Extra correct info is OK; no conflicts.
Return 'correct=True' only if it fully matches without contradictions.
Explain your reasoning first, then the label.
"""


correctness_llm = llm.with_structured_output(CorrectnessGrade, method="json_schema", strict=True)


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    msg = f"""QUESTION: {inputs['question']}
GROUND TRUTH ANSWER: {reference_outputs['answer']}
STUDENT ANSWER: {outputs.get('answer','')}"""
    grade = correctness_llm.invoke(
        [{"role": "system", "content": correctness_instructions},
         {"role": "user", "content": msg}]
    )
    return bool(grade["correct"])

# RELEVANCE: Response vs Input 

class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "Provide the score on whether the answer addresses the question"]


relevance_instructions = """You are a teacher grading a quiz.
You will be given a QUESTION and a STUDENT ANSWER.
Return relevant=True only if the answer addresses the question and helps the user.
Explain your reasoning first, then the label.
"""

relevance_llm = llm.with_structured_output(RelevanceGrade, method="json_schema", strict=True)


def relevance(inputs: dict, outputs: dict) -> bool:
    msg = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs.get('answer','')}"
    grade = relevance_llm.invoke(
        [{"role": "system", "content": relevance_instructions},
         {"role": "user", "content": msg}]
    )
    return bool(grade["relevant"])


#GROUNDEDNESS: Response vs Retrieved Docs

class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded:   Annotated[bool, ..., "True if the answer is fully supported by the facts"]


grounded_instructions = """You are a teacher grading groundedness.
You will be given FACTS (retrieved context) and a STUDENT ANSWER.
Return grounded=True only if all claims are supported by the FACTS (no hallucinations).
Explain your reasoning first, then the label.
"""


grounded_llm = llm.with_structured_output(GroundedGrade, method="json_schema", strict=True)


def groundedness(inputs: dict, outputs: dict) -> bool:
    docs = outputs.get("documents") or []
    doc_string = "\n\n".join(d.get("page_content","") for d in docs)
    msg = f"FACTS:\n{doc_string}\n\nSTUDENT ANSWER:\n{outputs.get('answer','')}"
    grade = grounded_llm.invoke(
        [{"role": "system", "content": grounded_instructions},
         {"role": "user", "content": msg}]
    )
    return bool(grade["grounded"])


# RETRIEVAL RELEVANCE: Retrieved Docs vs Input (question)

class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant:    Annotated[bool, ..., "True if retrieved docs are relevant to the question"]


retrieval_relevance_instructions = """You are grading if the retrieved FACTS are relevant to the QUESTION.
If ANY portion is semantically related, consider them relevant. Minor off-topic content is OK.
Explain your reasoning first, then the label.
"""


retrieval_relevance_llm = llm.with_structured_output(RetrievalRelevanceGrade, method="json_schema", strict=True)


def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    docs = outputs.get("documents") or []
    doc_string = "\n\n".join(d.get("page_content","") for d in docs)
    msg = f"QUESTION:\n{inputs['question']}\n\nFACTS:\n{doc_string}"
    grade = retrieval_relevance_llm.invoke(
        [{"role": "system", "content": retrieval_relevance_instructions},
         {"role": "user", "content": msg}]
    )
    return bool(grade["relevant"])

def target(inputs: dict) -> dict:
    q = inputs["question"]
    result = runnable.invoke({"question": q})


    answer = result.get("answer", "")

   
    docs = result.get("documents") or result.get("contexts") or result.get("chunks") or []
    if isinstance(docs, str):
        docs = [docs]
    documents = [{"page_content": str(d)} for d in docs]

    return {"answer": answer, "documents": documents}


experiment = client.evaluate(
    target,
    data=DATASET_NAME,  
    evaluators=[correctness, relevance, groundedness, retrieval_relevance],
    experiment_prefix="rag-baseline-langgraph",
    metadata={"grader_model": GRADER_MODEL},
)

print(" RAG eval complete. View results in LangSmith.")
try:
    print(experiment.to_pandas().head())
except Exception:
    pass
