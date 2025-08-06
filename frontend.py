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
            st.write("**Top Chunks (from DB):**")
            for chunk in result.get("top_chunks", []):
                st.write(f"-{chunk}")
                st.markdown("---")  # adds a line break to force visual separation
                st.write(response.json())  # TEMP: debug raw output

        else:
            st.error("Error: Could not get answer from API")
    else:
        st.warning("Please enter a question.")