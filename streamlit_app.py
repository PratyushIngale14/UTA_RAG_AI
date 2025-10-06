import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from pinecone import Pinecone
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 0. Configuration from Streamlit Secrets ---
INDEX_NAME = "uta-rag" 

# Set environment variables (keys are passed directly to classes below)
os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
PINECONE_REGION = st.secrets["PINECONE_REGION"] 

# --- 1. RAG Core Components Initialization ---
@st.cache_resource(show_spinner="Connecting to the RAG Knowledge Base...")
def initialize_rag():
    
    # Initialize the Pinecone client
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    
    # 1. Embeddings Model: Authenticated directly for robust connection
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        api_key=os.environ["GEMINI_API_KEY"] 
    )
    
    # 2. Vector Store Retriever: Querying side
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
    
    # 4. RAG Prompt Template (Includes instructions for Q&A, Notes, and Quiz)
    SYSTEM_TEMPLATE = (
        "You are a helpful and expert University Study Assistant specializing in Data Science Project Management. "
        "Your goal is to facilitate learning using *only* the retrieved context provided below. "
        "Maintain a professional and encouraging tone. "
        "If you cannot find the answer in the context, you must state: "
        "'The provided course material does not contain enough information on this specific topic.'\n\n"
        "**Specific Task Guidance:**\n"
        "1. **Answer** the user's question directly.\n"
        "2. If the user asks for 'notes' or 'summary', generate 3-5 concise, well-structured bullet points.\n"
        "3. If the user asks for a 'quiz', 'questions', or 'test', generate a 3-question multiple-choice quiz "
        "with answers based strictly on the context. Format the quiz clearly with options (A, B, C).\n\n"
        "**Retrieved Context:**\n{context}"
    )
    
    # Define the core RAG chain using LCEL (LangChain Expression Language)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_template(SYSTEM_TEMPLATE)
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# --- Image Path ---
# Use a static link structure that is guaranteed to work on Streamlit Cloud
# Replace YOUR_REPO_NAME if it's not UTA_RAG_AI
# The file name is "UTA Banner.png" (note the spaces)
GITHUB_RAW_PATH = "https://raw.githubusercontent.com/PratyushIngale14/UTA_RAG_AI/main/UTA%20Banner.png"

rag_chain = initialize_rag()

# --- 2. Streamlit UI and Chat Logic ---
st.set_page_config(page_title="UTA RAG Study Assistant", layout="wide")

# --- Custom Header/Branding (Display image) ---
# FIX: Using st.image with the direct, encoded GitHub raw URL
st.image(GITHUB_RAW_PATH, use_container_width=True) 

# New Main Title and Subject Subtitle
st.markdown(
    """
    # University of Texas at Arlington RAG Study Assistant
    ### Subject: Data Science Project Management
    """
)

# Initialize chat history (Updated welcome message and removed emoji)
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
                answer = rag_chain.invoke(prompt)
                
                st.markdown(answer)

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                error_message = f"An error occurred during generation. This may be due to a Streamlit Cloud resource limit or an API timeout. Error: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
