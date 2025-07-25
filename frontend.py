import streamlit as st 
import requests

st.title("Medical Insurance Q&A Assistant")

user_input = st.text_area("Ask a question about your insurance plan:", "")
if st.button("Submit"):
    if user_input.strip():
        response = requests.post(
            "http://127.0.0.1:8000/ask",
            json={"question": user_input}
        )
        if response.status_code == 200:
            result = response.json()
            st.write("**Top Chunks (from DB):**")
            for chunk in results.get("top_chunks", []):
                st.write(f"-{chunk}")
        else:
            st.error("Error: Could not get answer from API")
    else:
        st.warning("Please enter a question.")