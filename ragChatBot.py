from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
import pdfplumber
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


st.header("Saurabh's GenAI Joureny")

# creating Session
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_file_name" not in st.session_state:
    st.session_state.processed_file_name = None
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type=["pdf","csv","txt"])

#Extract content from PDF and chunk 
if file is not None:
    if st.session_state.processed_file_name != file.name:   
        #Extract text
        with pdfplumber.open(file) as dataFile:
            text=''
            for page in dataFile.pages:
                text+=page.extract_text() + '\n'
        #st.write(text)

        #split text into chunks
        text_splitter= RecursiveCharacterTextSplitter(
            separators=['\n\n','\n',' ','. ',''],
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks=text_splitter.split_text(text)
        #st.write(chunks)
        st.success("Chunked Successfully!!!")
        #print(len(chunks))
        #print(chunks[:2])
    
        
        with st.spinner("Loading local model and indexing vectors..."):
            
            # Uses the downloaded model from your cache!
            embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5"
            )
            # Store the finished database into the session state memory!
            st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
            
            # Record the file name so we know we've processed it
            st.session_state.processed_file_name = file.name

            
            #vector_store = FAISS.from_texts(chunks, embeddings)
            
            st.success("Successfully embedded and stored in FAISS Vector DB locally!")
        
    #user question

    user_input=st.text_input("Enter Question!")

        # generate Answer
        # Chain process
        #       question -> embeddings -> simililarity search -> sending results to LLM -> response
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
        
    retriever=st.session_state.vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k":3}
    )

        #define the LLM and prompts

    #llm = ChatGoogleGenerativeAI(
    #    model="gemini-2.5-flash", 
    #   google_api_key=GEMINI_API_KEY,
    #    temperature=0.3 # Low temperature = more factual, less creative
    #)
    
    llm = OllamaLLM(
            model="llama3.2",
            temperature=0.3
    )

        #provide the prompts
    prompt = ChatPromptTemplate.from_messages([
            ("system",
            "You are a helpful assistant answering questions about a PDF document.\n\n"
            "Guidelines:\n"
            "1. Provide complete, well-explained answers using the context below.\n"
            "2. Include relevant details, numbers, and explanations to give a thorough response.\n"
            "3. If the context mentions related information, include it to give fuller picture.\n"
            "4. Only use information from the provided context - do not use outside knowledge.\n"
            "5. Summarize long information, ideally in bullets where needed\n"
            "6. If the information is not in the context, say so politely.\n\n"
            "Context:\n{context}"),
            ("human", "{question}")
        ])

    chain = (
            {"context":retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    if user_input:
        response = chain.invoke(user_input)
        st.write(response)

