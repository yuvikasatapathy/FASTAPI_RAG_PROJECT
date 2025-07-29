from rag_langgraph import runnable

result = runnable.invoke({"text": "/Users/richa/Downloads/sample_medical_insurance.pdf"})
print(result)