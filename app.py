import streamlit as st
from legal_assistant import LegalAIAssistant

st.set_page_config(page_title="Legal AI Assistant", layout="wide")
st.title("⚖️ Legal AI Assistant")
st.write("Ask questions about legal provisions and get concise AI-generated answers with context.")

@st.cache_resource
def load_assistant():
    assistant = LegalAIAssistant()
    df = assistant.load_dataset("hf://datasets/Moataz88Saad/ledgar_qa_retrieval/dataset.parquet")
    df["embeddings"] = list(assistant.get_embeddings(df["provision"].tolist()))
    return assistant, df

assistant, df = load_assistant()

question = st.text_input("Enter your legal question:")

if st.button("Get Answer") and question:
    with st.spinner("Generating answer..."):
        answer, context = assistant.generate_answer(question, df["provision"].tolist(), df["embeddings"].tolist())
        st.subheader("Answer")
        st.write(answer)
        st.subheader("Supporting Context")
        st.write(context)
