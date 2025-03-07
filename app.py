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

# Load environment variables
load_dotenv()

# Get API key
groq_api_key = st.secrets["GROQ_API_KEY"]

# Ensure API key is available
if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env file. Please check your setup.")
    st.stop()

# Initialize chat model
llm = init_chat_model("llama3-8b-8192", model_provider="groq", api_key=groq_api_key)

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
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Ensure the directory exists
        pdf_dir = r"./research_papers"
        if not os.path.exists(pdf_dir):
            st.error(f"Directory '{pdf_dir}' not found. Please check the path.")
            return

        st.session_state.loader = PyPDFDirectoryLoader(pdf_dir)
        st.session_state.docs = st.session_state.loader.load()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs[:5])

        # Create FAISS vector database
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)
        st.success("Vector DB is ready!")

# Streamlit UI
st.title("RAG Chatbot with FAISS")

user_prompt = st.text_input("Enter your query from the research paper")

if st.button("Document Embedding"):
    create_vector_embedding()

if user_prompt:
    if "vectors" not in st.session_state:
        st.error("Please create the vector database first by clicking 'Document Embedding'.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        rag_chain = create_retrieval_chain(retriever, document_chain)

        # Measure response time
        start_time = time.process_time()
        response = rag_chain.invoke({'input': user_prompt})
        elapsed_time = time.process_time() - start_time

        st.write(f"Response time: {elapsed_time:.2f} seconds")
        st.write(response.get('answer', "No response generated."))

        with st.expander("Document similarity search"):
            for i, doc in enumerate(response.get('context', [])):
                st.write(doc.page_content)
                st.write('-------------')
