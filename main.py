import os
import time
import pickle
import streamlit as st
from dotenv import load_dotenv

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub

# Load environment variables (for Hugging Face API key)
load_dotenv()

# Setup UI
st.title("🧠 RockyBot: News Research Tool (Chroma + HuggingFace)")
st.sidebar.title("Enter News URLs")

urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("Process URLs")

PERSIST_DIR = "chroma_store"
main_placeholder = st.empty()

# Define LLM
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",  # Or other Hugging Face-supported LLM
    model_kwargs={"temperature": 0.5, "max_length": 512},
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

# When "Process URLs" is clicked
if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("🔄 Loading content from URLs...")
    data = loader.load()

    # Split content into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    main_placeholder.text("✂️ Splitting text into chunks...")
    docs = splitter.split_documents(data)

    # Create embeddings using Hugging Face
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Store in Chroma DB
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    vectorstore.persist()

    main_placeholder.success("✅ URLs processed and stored successfully.")

# Question input
query = main_placeholder.text_input("🔍 Ask your question:")

if query:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever()

    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
    result = chain({"question": query}, return_only_outputs=True)

    st.header("📜 Answer")
    st.write(result["answer"])

    if result.get("sources"):
        st.subheader("🔗 Sources")
        for src in result["sources"].split("\n"):
            st.write(src)
