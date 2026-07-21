from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
import streamlit as st
import pdfplumber
import os
from dotenv import load_dotenv
import hashlib
import uuid


load_dotenv()

st.header("GenAI Journey - RAG ChatBot")

# Session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_file_name" not in st.session_state:
    st.session_state.processed_file_name = None
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]


with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type=["pdf"])

if file is not None:
    pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    # Generate a unique namespace per uploaded file
    namespace = f"{st.session_state.session_id}_{hashlib.md5(file.name.encode()).hexdigest()[:8]}"
    #namespace = hashlib.md5(file.name.encode()).hexdigest()[:12]

    index_name = "chatbot"
    existing_indexes = [index.name for index in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    if st.session_state.processed_file_name != file.name:
        # ---- Step 1: Extract + chunk ----
        with st.spinner("Chunking in progress..."):
            with pdfplumber.open(file) as dataFile:
                text = ''
                for page in dataFile.pages:
                    text += page.extract_text() + '\n'

            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', ' ', '. ', ''],
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_text(text)
        st.success(f"Chunked into {len(chunks)} pieces successfully!")

        # ---- Step 2: Embedding (with progress bar) ----
        with st.spinner("Embedding in progress..."):
            progress_bar = st.progress(0)
            chunk_vectors = []
            for i, chunk in enumerate(chunks):
                chunk_vectors.append(embeddings.embed_query(chunk))
                progress_bar.progress((i + 1) / len(chunks))
            progress_bar.empty()
        st.success("Embedding complete!")

        # ---- Step 3: Insert into Pinecone ----
        with st.spinner("Vector insert in progress..."):
            index = pc.Index(index_name)
            try:
                index.delete(delete_all=True, namespace=namespace)  # clean slate
            except Exception:
                pass  # namespace may not exist yet on first upload — safe to ignore
            vector_data = [(str(i), chunk_vectors[i], {"text": chunks[i]}) for i in range(len(chunks))]
            index.upsert(vectors=vector_data, namespace=namespace)
        st.success("Successfully stored in Pinecone!")

        st.session_state.vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=namespace)
        st.session_state.processed_file_name = file.name

    else:
        # Same file as before — just reconnect, don't chunk/embed/upsert again
        st.session_state.vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            namespace=namespace
        )

    user_input = st.text_input("Enter Question!")

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    retriever = st.session_state.vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.environ['GEMINI_API_KEY'],
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant answering questions about a PDF document.\n\n"
         "Guidelines:\n"
         "1. Provide complete, well-explained answers using the context below.\n"
         "2. Include relevant details, numbers, and explanations to give a thorough response.\n"
         "3. If the context mentions related information, include it to give a fuller picture.\n"
         "4. Only use information from the provided context - do not use outside knowledge.\n"
         "5. Summarize long information, ideally in bullets where needed.\n"
         "6. If the information is not in the context, say so politely.\n\n"
         "Context:\n{context}"),
        ("human", "{question}")
    ])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    if user_input:
        response = chain.invoke(user_input)
        st.write(response)