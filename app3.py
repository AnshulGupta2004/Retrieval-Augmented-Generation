import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import tempfile
import pyttsx3
import speech_recognition as sr
import io
from PyPDF2 import PdfReader
import pickle
import base64
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(temperature=0,model_name="mixtral-8x7b-32768", groq_api_key=os.environ["GROQ_API_KEY"])

# Initialize Ollama embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Initialize speech recognition
recognizer = sr.Recognizer()

def process_documents(docs, is_url=False):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = []
    for doc in docs:
        if is_url:
            loader = WebBaseLoader(doc)
            pages = loader.load()
        else:
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, doc.name)
            with open(temp_path, "wb") as f:
                f.write(doc.getvalue())
            loader = PyPDFLoader(temp_path)
            pages = loader.load_and_split(text_splitter)
        texts.extend(pages)
    
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # Save embeddings
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)
    
    return vectorstore

def load_vectorstore():
    if os.path.exists("vectorstore.pkl"):
        with open("vectorstore.pkl", "rb") as f:
            return pickle.load(f)
    return None

def get_audio_input():
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.error("Sorry, I couldn't understand the audio.")
        return None
    except sr.RequestError:
        st.error("Sorry, there was an error processing the audio.")
        return None

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.save_to_file(text, 'output.mp3')
    engine.runAndWait()

def highlight_pdf(pdf_path, page_num, start_char, end_char):
    # This is a placeholder function. Implementing PDF highlighting is complex and
    # may require additional libraries or a different approach.
    # For now, we'll just return the original PDF content
    with open(pdf_path, 'rb') as file:
        base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    return base64_pdf

def clean_url_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

st.set_page_config(layout="wide")
st.title("Chat with Your Documents")

# Sidebar for document upload
with st.sidebar:
    st.subheader("Upload your documents")
    doc_type = st.radio("Choose document type:", ("PDF", "URL"))
    if doc_type == "PDF":
        docs = st.file_uploader("Choose your PDF files", type="pdf", accept_multiple_files=True)
    else:
        urls = st.text_area("Enter URLs (one per line)")
        docs = urls.split('\n') if urls else []

    if st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            vectorstore = process_documents(docs, is_url=(doc_type == "URL"))
            st.session_state.vectorstore = vectorstore
            st.success("Documents processed successfully!")

# Main area with two columns
col1, col2 = st.columns([1, 1])

# First column: Document viewer
with col1:
    st.subheader("Document Viewer")
    if 'docs' in locals() and docs:
        if doc_type == "PDF":
            selected_doc = st.selectbox("Select a PDF to view", [doc.name for doc in docs])
            selected_doc_file = next((doc for doc in docs if doc.name == selected_doc), None)
            if selected_doc_file:
                pdf_reader = PdfReader(io.BytesIO(selected_doc_file.getvalue()))
                page_num = st.number_input("Page number", min_value=1, max_value=len(pdf_reader.pages), value=1)
                page = pdf_reader.pages[page_num - 1]
                st.text_area("PDF Content", page.extract_text(), height=400)
        else:
            selected_url = st.selectbox("Select a URL to view", docs)
            if selected_url:
                cleaned_content = clean_url_content(selected_url)
                st.text_area("URL Content", cleaned_content, height=400)

# Second column: Chat interface
with col2:
    st.subheader("Chat with Documents")
    
    # Initialize conversation history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display conversation history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input
    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        user_input = st.text_input("Your question:")
    with col2:
        if st.button("🎤"):
            user_input = get_audio_input()
            if user_input:
                st.write(f"You said: {user_input}")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)

        vectorstore = st.session_state.get('vectorstore') or load_vectorstore()
        
        if vectorstore:
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=memory
            )
            
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = conversation_chain({"question": user_input})
                        st.markdown(response['answer'])
                        
                        # Generate audio output
                        text_to_speech(response['answer'])
                        st.audio('output.mp3')
                        
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})
        else:
            st.error("Please upload and process documents first.")

    # Add a button to clear the conversation
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.experimental_rerun()

# CSS to make the document viewer column fixed
st.markdown("""
    <style>
        .fixed-content { position: fixed; }
    </style>
    """, unsafe_allow_html=True)
st.markdown('<div class="fixed-content">', unsafe_allow_html=True)