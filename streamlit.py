import langchain
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import streamlit as st
import base64

# TO RUN: streamlit run streamlit.py

llm = Ollama(model="llama3:latest")

@st.cache_resource
class PdfGpt():
    def __init__(self, file_path):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        chunks = text_splitter.split_documents(documents=PyMuPDFLoader(file_path=file_path).load())
        
        embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device':'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vectorstore = FAISS.from_documents(chunks, embedding_model)
        vectorstore.save_local("vectorstore")
        
        template = """
        ### System:
        You are a respectful and honest assistant. You have to answer the user's questions using only the paper, or pdf, or document context \
        provided to you. If you don't know the answer, just say you don't know. Don't try to make up an answer.

        ### Context:
        {context}

        ### User:
        {question}

        ### Response:
        """

        self.hey = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={'prompt': PromptTemplate.from_template(template)}
        )

def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    return pdf_display

st.set_page_config(layout="wide")
st.title("Analyze Your Paper")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with open("uploaded_pdf.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    oracle = PdfGpt("uploaded_pdf.pdf")
    
    col1, col2 = st.columns(2)

    with col1:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("What's up?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response = oracle.hey({'query': prompt})
                result = response['result']
                st.markdown(result)

            st.session_state.messages.append({"role": "assistant", "content": result})

    with col2:
        st.markdown(show_pdf("uploaded_pdf.pdf"), unsafe_allow_html=True)