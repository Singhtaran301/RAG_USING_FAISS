import streamlit as st
import langchain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import time
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
# Load environment variables locally
load_dotenv()

# Get API key (first try secrets, then .env)
groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

# Ensure API key is available
if not groq_api_key:
    st.error("❌ GROQ_API_KEY is missing! Add it in Streamlit Secrets (not .env) for cloud deployment.")
    st.stop()

# Initialize chat model
llm = ChatGroq(model_name="llama3-8b-8192", api_key=groq_api_key)

# Define prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the following context:
    <context>
    {context}
    </context>
    Question: {input}
    """
)

# Function to create vector embeddings
def create_vector_embedding():
    if "vectors" not in st.session_state:
        # Retrieve the uploaded file
        if "uploaded_file" not in st.session_state:
            st.error("Please upload a PDF file first.")
            return
        
        uploaded_file = st.session_state.uploaded_file

        # Save file temporarily
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load PDF properly
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_docs = text_splitter.split_documents(docs)

        # Create FAISS vector database
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.vectors = FAISS.from_documents(final_docs, embeddings)

        st.success("✅ Vector DB is ready!")

# Streamlit UI
st.title("RAG Chatbot with FAISS")

# File uploader with persistence
uploaded_file = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])
if uploaded_file:
    st.session_state.uploaded_file = uploaded_file  # Store file persistently

if st.button("Document Embedding"):
    create_vector_embedding()

# Handle user queries
user_prompt = st.text_input("Enter your query from the research paper")

if user_prompt:
    if "vectors" not in st.session_state:
        st.error("⚠️ Please create the vector database first by clicking 'Document Embedding'.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        rag_chain = create_retrieval_chain(retriever, document_chain)

        # Measure response time
        start_time = time.process_time()
        response = rag_chain.invoke({'input': user_prompt})
        elapsed_time = time.process_time() - start_time

        st.write(f"⏱ Response time: {elapsed_time:.2f} seconds")
        st.write(response.get('answer', "No response generated."))

        with st.expander("📄 Document similarity search"):
            for i, doc in enumerate(response.get('context', [])):
                st.write(doc.page_content)
                st.write('-------------')