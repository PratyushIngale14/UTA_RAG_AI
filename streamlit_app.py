import streamlit as st
import os
import time

# --- LangChain + Google + Pinecone ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone
import google.generativeai as genai  

# --- CONFIG ---
INDEX_NAME = "uta-rag"

# --- Secret Key Validation ---
if "GEMINI_API_KEY" not in st.secrets or "PINECONE_API_KEY" not in st.secrets:
    st.error("Missing required API keys in Streamlit Secrets! Please add GEMINI_API_KEY and PINECONE_API_KEY.")
else:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

os.environ["NO_GCE_CHECK"] = "true"  # Prevents GCP metadata lookup issues

# --- Initialize Gemini Client ---
GEMINI_CLIENT = None
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    GEMINI_CLIENT = genai
except Exception as e:
    st.error(f"Failed to initialize Gemini Client: {e}")

# --- Embeddings Wrapper ---
class StreamlitEmbeddings:
    """Wrapper to call Google GenAI embeddings"""
    def embed_query(self, text):
        if GEMINI_CLIENT is None:
            raise Exception("Gemini Client not initialized.")
        try:
            result = GEMINI_CLIENT.embed_content(
                model="models/text-embedding-004",
                content=text,
            )
            # Defensive extraction
            if "embedding" in result:
                return result["embedding"]
            else:
                raise Exception("Unexpected embedding format returned from Gemini.")
        except Exception as e:
            raise Exception(f"Error embedding content: {e}")

# --- RAG Chain Initialization ---
@st.cache_resource(ttl=3600, max_entries=1)
def initialize_rag_chain(_embeddings_client):
    try:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        indexes = [i["name"] for i in pc.list_indexes()]

        if INDEX_NAME not in indexes:
            raise Exception(f"Index '{INDEX_NAME}' not found in your Pinecone account.")

        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=_embeddings_client,
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.3,
            google_api_key=os.environ["GEMINI_API_KEY"],
            timeout=180,
        )

        SYSTEM_TEMPLATE = (
            "You are a helpful University Study Assistant specializing in Data Science Project Management.\n"
            "Use ONLY the retrieved context provided below. If missing, reply:\n"
            "'The provided course material does not contain enough information on this specific topic.'\n\n"
            "**Task Guidance:**\n"
            "1. If asked for 'notes/summary', give 3–5 concise bullet points.\n"
            "2. If asked for 'quiz/questions/test', make 3 multiple-choice Qs (A,B,C options) with answers.\n\n"
            "**Retrieved Context:**\n{context}"
        )

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | ChatPromptTemplate.from_template(SYSTEM_TEMPLATE)
            | llm
            | StrOutputParser()
        )

        return rag_chain

    except Exception as e:
        st.error(f"RAG initialization failed: {e}")
        return None

# --- Streamlit UI ---
st.set_page_config(page_title="UTA RAG Study Assistant", layout="wide")
st.image("UTA Banner.png", use_container_width=True)
st.title("University of Texas at Arlington RAG Study Assistant")
st.caption("Subject: Data Science Project Management")

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize only once
if st.session_state.rag_chain is None and GEMINI_CLIENT is not None:
    with st.spinner("Connecting to RAG system..."):
        st.session_state.rag_chain = initialize_rag_chain(StreamlitEmbeddings())
        if st.session_state.rag_chain:
            st.success("RAG system ready! ")

# --- Chat Interface ---
if st.session_state.rag_chain:
    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! I am your UTA Study Assistant. Try asking:\n"
                       "- 'Summarize the four phases of the project life cycle.'\n"
                       "- 'Create a 3-question quiz on scope creep.'"
        })

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask me about your course..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = st.session_state.rag_chain.invoke(prompt)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Query failed: {e}")
else:
    st.warning("RAG system not available. Check API keys and index configuration.")
