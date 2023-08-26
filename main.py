import sys
import os
import tempfile
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
from streamlit_extras.buy_me_a_coffee import button
button(username="Woo Hyeon Her", floating=True, width=221)
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

load_dotenv()

# Title
st.title("ChatPDF")
st.write("---")

# Open AI KEY
openai_key = st.text_input('OPEN_AI_API_KEY', type="password")

# File Upload
uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])
st.write("---")


def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages


# What happens when uploading is completed
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=300,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(pages)

    # Embedding
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key)

    # Load it into Chroma
    db = Chroma.from_documents(texts, embeddings_model)

    # Question
    st.header("Asking questions to PDF!!")
    question = st.text_input('Write your question')
    if st.button('Ask your question'):
        with st.spinner('Coming up with the answer...'):
            llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                             temperature=0, openai_api_key=openai_key)
            qa_chain = RetrievalQA.from_chain_type(
                llm, retriever=db.as_retriever())
            result = qa_chain({"query": question})
            st.write(result["result"])
