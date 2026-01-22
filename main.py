import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_classic.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Initialize Streamlit app
st.title("News Research Tool")

st.sidebar.title("News Article URLs")

# collect up to 3 URLs from the user
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url.strip():
        urls.append(url.strip())

process_urls = st.sidebar.button("Process URLs")


main_placeholder = st.empty()

query = st.text_input("Enter your research question here:")

ask_btn = st.button("Ask")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Vectorstore initialized as None and populated when processing URLs or loading from disk
vectorstore = None


if process_urls and urls:
    # Initialize the URL loader with the provided URLs
    try:
        loader = WebBaseLoader(urls)
        main_placeholder.text("Loading and processing documents...")
        data = loader.load()

        main_placeholder.text("Splitting documents into chunks...")
        # Split the documents into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', ' ', ''],
            chunk_size=1000,
            chunk_overlap=150,
        )
        docs = text_splitter.split_documents(data)
    except Exception as e:
        print(e)
        main_placeholder.text(f"Error loading URLs: {e}")
        st.stop()

    main_placeholder.text("Creating embeddings and saving to FAISS vector store...")
    # Create Embeddings and save to FAISS vector store
    vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local("faiss_index")
    st.success("✅ FAISS index saved!")

# Load existing FAISS index to avoid re-processing URLs on app rerun
elif os.path.exists("faiss_index"):
    main_placeholder.text("Loading FAISS vector store...")
    vectorstore = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    main_placeholder.text("✅ FAISS index loaded!")


if ask_btn and query:
    if vectorstore is None:
        st.warning("No FAISS index found. Please enter URLs and process them to create the index!")
        st.stop()

    chain = RetrievalQAWithSourcesChain.from_llm(llm=ChatGoogleGenerativeAI(model="models/gemini-flash-lite-latest", temperature=0.3), retriever=vectorstore.as_retriever())

    result = chain.invoke({"question": query})
    answer = result["answer"]

    st.header("Answer:")
    st.write(answer)

    st.header("Sources:")
    sources_text = result.get("sources", "").strip()
    
    if sources_text:
        for source in sources_text.split("\n"):
            source = source.strip()
            if source:
                st.write(f"- {source}")
    else:
        st.write("No sources found.")