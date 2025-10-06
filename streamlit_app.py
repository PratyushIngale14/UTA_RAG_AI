import streamlit as st
import os
import time 
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone

# --- 0. Configuration from Streamlit Secrets ---
INDEX_NAME = "uta-rag" 

# Set environment variables (Keys are passed directly to classes below)
os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

# === ABSOLUTE FINAL FIX: DISABLE GCE METADATA SERVER LOOKUP ===
# This prevents the SDK from wasting 60 seconds trying to fetch credentials
# from a server that doesn't exist in the Streamlit environment.
os.environ["NO_GCE_CHECK"] = "true" 
# =============================================================

# --- 1a. Embeddings and Retriever Initialization ---
def initialize_embeddings_and_retriever():
    
    # 1. Pinecone Client & Embeddings 
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        api_key=os.environ["GEMINI_API_KEY"] 
    )
    
    # 2. Vector Store Retriever: Connect to the EXISTING index 
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME, 
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    return embeddings, retriever

# --- 1b. RAG Chain Initialization (Uses components from 1a) ---
def initialize_rag_chain(embeddings, retriever):
    
    # 3. LLM (Using the fast, free-tier model)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite", 
        temperature=0.3,
        google_api_key=os.environ["GEMINI_API_KEY"],
        # FINAL TIMEOUT FIX: Set a large timeout for the initial connection and subsequent requests
        # This overrides the default 60s Streamlit/Google SDK timeout.
        timeout=180 
    )
    
    # 4. RAG Prompt Template
    SYSTEM_TEMPLATE = (
        "You are a helpful and expert University Study Assistant specializing in Data Science Project Management. "
        "Your goal is to facilitate learning using *only* the retrieved context provided below. "
        "If you cannot find the answer in the context, you must state: "
        "'The provided course material does not contain enough information on this specific topic.'\n\n"
        "**Specific Task Guidance:**\n"
        "1. **Answer** the user's question directly.\n"
        "2. If the user asks for 'notes' or 'summary', generate 3-5 concise, well-structured bullet points.\n"
        "3. If the user asks for 'quiz', 'questions', or 'test', generate a 3-question multiple-choice quiz "
        "with answers based strictly on the context. Format the quiz clearly with options (A, B, C).\n\n"
        "**Retrieved Context:**\n{context}"
    )
    
    # Define the core RAG chain using LCEL
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_template(SYSTEM_TEMPLATE)
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


# --- UI Setup and Logic ---

st.set_page_config(page_title="UTA RAG Study Assistant", layout="wide")

# Image and Title 
st.image("UTA Banner.png", use_container_width=True) 

st.markdown(
    """
    # University of Texas at Arlington RAG Study Assistant
    ### Subject: Data Science Project Management
    """
)

# --- App State Management ---
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False


# --- Main Logic Flow: Progressive Initialization ---
if not st.session_state.initialized:
    # Attempt to initialize components if they haven't been yet
    with st.spinner("Initial Cold Start: Waking up RAG and Gemini services. **This is the final connection attempt. Please wait.**"):
        try:
            # Step 1: Initialize Embeddings and Retriever (The slow part)
            if st.session_state.retriever is None:
                embeddings, retriever = initialize_embeddings_and_retriever()
                st.session_state.embeddings = embeddings
                st.session_state.retriever = retriever
            
            # Step 2: Initialize the RAG Chain (The LLM part)
            if st.session_state.rag_chain is None and st.session_state.retriever is not None:
                st.session_state.rag_chain = initialize_rag_chain(
                    st.session_state.embeddings, 
                    st.session_state.retriever
                )
                st.session_state.initialized = True
                st.success("RAG system successfully initialized! Ask your first question.")
                st.rerun() # Rerun to update the UI cleanly
                
        except Exception as e:
            # This captures the 504 and prompts a retry.
            st.error("RAG system initialization failed due to connection timeout. **Please try refreshing the app now (Ctrl+R/F5)**.")
            # st.exception(e) # Do not display exception repeatedly, just the error message

# --- Chat Interface ---
if st.session_state.rag_chain is not None and st.session_state.initialized:
    # If initialization succeeded, proceed with chat logic
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": 
            "Hello! I am your UTA Study Assistant. Ask me anything about your Data Science Project Management material, or try: "
            "\n\n- 'Summarize the four phases of the project life cycle.'"
            "\n- 'Create a 3-question quiz on scope creep.'"
        })

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Enter your question, or ask for notes/quiz:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call the RAG chain
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base and generating answer..."):
                try:
                    # Invoke the RAG chain
                    answer = st.session_state.rag_chain.invoke(prompt)
                    
                    st.markdown(answer)

                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    error_message = f"An error occurred during query. Error: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
else:
    # This warning will appear if the initial load failed, prompting the manual retry.
    st.warning("System is completing its initial setup. Please wait and refresh the page now to force retry.")