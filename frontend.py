import streamlit as st 
import requests
import time 

st.title("Medical Insurance Q&A Assistant")

user_input = st.text_area("Ask a question about your insurance plan:", "")
if st.button("Submit"):
    if user_input.strip():
        response = requests.post(
            "http://127.0.0.1:8003/ask",
            json={"question": user_input}
        )
        if response.status_code == 200:
            result = response.json()
            st.write("**Answer:**")
            with st.empty():
                streamed_text = ""
                for word in result.get("answer", "").split():
                    streamed_text += word + " "
                    st.success(streamed_text)
                    time.sleep(0.05)
            st.markdown("**Top Chunks (from DB):**")
            seen_chunks = set()
            for chunk in result.get("top_chunks", []):
                if isinstance(chunk, list):
                    for inner in chunk: 
                        if inner not in seen_chunks:
                            st.markdown(f"-{inner}")
                            seen_chunks.add(inner)
                else:
                    if chunk not in seen_chunks:  
                        st.markdown(f"-{chunk}")
                        seen_chunks.add(chunk)
                #st.markdown("---")  # adds a line break to force visual separation
                

        else:
            st.error("Error: Could not get answer from API")
    else:
        st.warning("Please enter a question.")