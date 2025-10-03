# UTA_RAG_AI
University RAG Study Assistant (Project Management)
This is a free, open-source Retrieval-Augmented Generation (RAG) chatbot designed to act as an expert tutor on specific university course materials. It uses a Google Gemini Large Language Model (LLM) and a Pinecone Vector Database to provide grounded, accurate answers, summaries, and quizzes based only on the uploaded textbook (PM_TEXTBOOK.pdf).

✨ Features
Grounded Q&A: Answers are synthesized from the course material (indexed in Pinecone), eliminating LLM hallucination.

Quiz Generation: Responds to requests like "Create a 3-question quiz on risk management."

Study Notes: Generates concise, structured bullet-point summaries of complex topics.

Free and Open Access: Built entirely on the Gemini API Free Tier and Pinecone Starter Plan.

Cloud-Based: Deployed on Streamlit Cloud, requiring no local setup for users.

🛠️ Technology Stack
Component	Technology	Role
Frontend	Streamlit (Python)	Provides the interactive chat interface and deployment framework.
LLM/Embeddings	Google Gemini 2.5 Flash-Lite	The generative model for responses and the embedding model for vector conversion.
RAG Orchestration	LangChain (Python)	Manages the workflow: retrieval of context and prompting the LLM.
Vector Database	Pinecone (Serverless)	Cloud-hosted vector store for secure, scalable storage of vectorized textbook data.
PDF Processing	PyPDFLoader / RecursiveCharacterTextSplitter	Handles reading and chunking the course PDF.

Export to Sheets
🚀 Deployment & Setup (For Replication)
To run your own version of this chatbot with different data, follow these two phases:

Phase 1: Indexing the Knowledge Base (One-Time Setup)
This step must be run using a powerful environment like Google Colab to handle the heavy data processing and API calls, saving the vectors to the cloud database.

Get API Keys: Obtain your Gemini API Key and a Pinecone API Key (from a free Starter plan).

Run Colab Script: Upload your target PDF (e.g., PM_TEXTBOOK.pdf) to your Colab notebook. Run the cells in the provided index_documents.ipynb script, making sure to replace the placeholder API keys with your actual values in the configuration section.

The script handles document loading, splitting into ~1000-character chunks, embedding via Gemini, and batch-uploading to the Pinecone index (uta-rag).

Phase 2: Deploying the Chatbot Application
Repository Structure: Push the following files to a public GitHub repository:

requirements.txt

streamlit_app.py

Streamlit Cloud Secrets: Go to Streamlit Cloud and select "New App". Before deploying, configure your Secrets with the exact names used in the application:

Ini, TOML

# .streamlit/secrets.toml
GEMINI_API_KEY="your_full_gemini_api_key"
PINECONE_API_KEY="your_full_pinecone_api_key"
PINECONE_REGION="us-east-1" # Use the region you created the index in
Launch: Deploy the application, pointing to streamlit_app.py. The app will load, connect to your Pinecone index, and be ready to answer questions.

📁 Repository Structure
university-chatbot-rag/
├── .gitignore          # Ignores local environment files (e.g., venv, .env)
├── README.md           # This file
├── requirements.txt    # Lists all necessary Python dependencies for deployment
├── index_documents.ipynb # The Python Notebook used for the one-time data indexing
└── streamlit_app.py    # The main Streamlit web application code (the chatbot)