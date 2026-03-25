import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime
from langchain_classic.chains.question_answering import load_qa_chain


def get_pdf_text(pdf_docs):
    text = "    "
    for pdf in pdf_docs:
        pdf_render = PdfReader(pdf)
        for page in pdf_render.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text, model_name="Google Ai"):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks, model_name, api_key=None):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", api_key=api_key
    )
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


def get_conversational_chain(model_name, vector_store=None, api_key=None):
    if model_name == "Google AI":
        prompt_template = """
            Answer the question from the provided context. Make sure to answer the question based on the context provided. If the answer is not in the context provided, then say that ```I don't know the answer.``` 
            context: {context}
            question: {question}
            
            Answer: 
        """
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key)
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain


# take user input
def user_input(user_question, model_name, api_key, pdf_docs, conversation_history):
    if api_key is None or pdf_docs is None:
        st.warning("Please provide API key and upload PDF documents")
        return
    text_chuncks = get_text_chunks(get_pdf_text(pdf_docs), model_name)
    vector_store = get_vector_store(text_chuncks, model_name, api_key)
    user_question_output = ""
    response_output = ""

    if model_name == "Google AI":
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", api_key=api_key
        )
        new_db = FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True
        )
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain(
            model_name="Google AI", vector_store=new_db, api_key=api_key
        )
        response = chain.run(
            input_documents=docs, question=user_question, return_only_outputs=True
        )
        user_question_output = user_question
        response_output = (
            response
            if isinstance(response, str)
            else response.get("output_text", str(response))
        )
        pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []
        conversation_history.append(
            {
                "question": user_question,
                "answer": response_output,
                "pdf_names": ", ".join(pdf_names),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        st.markdown(
            f"""
            <div style="background-color: #f0f2f6; color: black; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                <p><strong>Question:</strong> {user_question}</p>
                <p><strong>Answer:</strong> {response_output}</p>
                <p><strong>PDFs Used:</strong> {", ".join(pdf_names)}</p>
                <p><strong>Timestamp:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Instead of deleting and modifying the array directly, we just loop the previous chats
        for chat in reversed(conversation_history[:-1]):
            st.markdown(
                f"""
                <div style="background-color: #f0f2f6; color: black; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                    <p><strong>Question:</strong> {chat.get("question")}</p>
                    <p><strong>Answer:</strong> {chat.get("answer")}</p>
                    <p><strong>Timestamp:</strong> {chat.get("timestamp")}</p>
                    <p><strong>PDFs Used:</strong> {chat.get("pdf_names")}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if len(st.session_state.conversation_history) > 0:
            df = pd.DataFrame(st.session_state.conversation_history)
            st.download_button(
                label="Download Conversation History",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="conversation_history.csv",
                mime="text/csv",
            )
            st.toast("Question answered successfully! ✅")


# main Entry point function
def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs (v1) :books:")

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    model_name = st.sidebar.radio("Select the Model:", ("Google AI"))

    api_key = None

    if model_name == "Google AI":
        api_key = st.sidebar.text_input("Enter your Google API Key:")
        st.sidebar.markdown("Click [here](https://ai.google.dev/) to get an API key.")

        if not api_key:
            st.sidebar.warning("Please enter your Google API Key to proceed.")
            return

    with st.sidebar:
        st.title("Menu:")

        col1, col2 = st.columns(2)

        reset_button = col2.button("Reset")
        clear_button = col1.button("Rerun")

        if reset_button:
            st.session_state.conversation_history = []  # Clear conversation history
            st.session_state.user_question = None  # Clear user question input

            api_key = None  # Reset Google API key
            pdf_docs = None  # Reset PDF document

        else:
            if clear_button:
                if "user_question" in st.session_state:
                    st.warning("The previous query will be discarded.")
                    st.session_state.user_question = ""  # Temizle
                    if len(st.session_state.conversation_history) > 0:
                        st.session_state.conversation_history.pop()  # Son sorguyu kaldır
                else:
                    st.warning("The question in the input will be queried again.")

        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True,
        )
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    st.success("Done")
            else:
                st.warning("Please upload PDF files before processing.")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(
            user_question,
            model_name,
            api_key,
            pdf_docs,
            st.session_state.conversation_history,
        )
        st.session_state.user_question = ""  # Clear user question input


if __name__ == "__main__":
    main()
