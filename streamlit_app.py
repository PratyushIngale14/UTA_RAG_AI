import streamlit as st
import os
import time # Added for potential delays
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone

# --- 0. Configuration from Streamlit Secrets ---
INDEX_NAME = "uta-rag" 
os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

# --- 1. Core RAG Chain Function (Optimized for Stability) ---
# NOTE: The @st.cache_resource is critical for stability and performance after the first successful load.
@st.cache_resource(ttl="1h", max_entries=1)
def initialize_rag_chain():
    st.info("Initializing RAG system components...")
    
    # 1. Pinecone Client & Embeddings (Keys are passed directly for stability)
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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) 

    # 3. LLM (Using the fast, free-tier model)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite", 
        temperature=0.3,
        google_api_key=os.environ["GEMINI_API_KEY"] 
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
    
    st.success("RAG system successfully initialized!")
    return rag_chain

# --- UI Setup and Logic ---

st.set_page_config(page_title="UTA RAG Study Assistant", layout="wide")

# Image and Title 
# FIX: Changed 'use_column_width' to 'use_container_width' to resolve deprecation warning.
st.image("UTA Banner.png", use_container_width=True) 

st.markdown(
    """
    # University of Texas at Arlington RAG Study Assistant
    ### Subject: Data Science Project Management
    """
)

# --- App State Management ---
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# --- Main Logic Flow ---
if st.session_state.rag_chain is None:
    # If the chain hasn't been initialized yet, try to initialize it
    with st.spinner("Initial Cold Start: Waking up RAG and Gemini services. This may take up to 60 seconds..."):
        try:
            # We call the cached resource function here
            st.session_state.rag_chain = initialize_rag_chain()
        except Exception as e:
            st.error("RAG system initialization failed due to connection error. Please try refreshing the app in 1 minute.")
            st.exception(e) # Show the full traceback for debugging

# --- Chat Interface ---
if st.session_state.rag_chain is not None:
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
                    # This error is now captured and displayed cleanly
                    error_message = f"An error occurred during query. Error: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
else:
    # If the initial load failed, prompt user to refresh
    st.warning("Please wait and refresh the page in a few minutes. The system is completing its initial setup.")