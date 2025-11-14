📘 ASK NOVA – RAG + Web Search + Groq Chatbot

ASK NOVA is an advanced AI assistant built using Retrieval-Augmented Generation (RAG), Web Search Tools, and Groq LLMs to deliver accurate, context-aware responses.

It supports:

📄 PDF, DOCX, TXT document ingestion

🔍 Arxiv, Wikipedia, DuckDuckGo web search

🧠 Chroma vector database

💬 Conversational memory

⚡ Ultra-fast Groq LLM responses

🔥 Hybrid mode → RAG + Web Search combined

🚀 Full Streamlit UI

⭐ Features
✅ 1. RAG (Retrieval-Augmented Generation)

Upload documents

Embed them locally using HuggingFace embeddings

Store vectors in Chroma DB

Retrieve relevant chunks during chat

✅ 2. Web Search Tools

ASK NOVA can search the internet using:

Arxiv

Wikipedia

DuckDuckGo search

✅ 3. Hybrid Mode (RAG + Web Search)

NOVA combines:

Context from your uploaded documents

Latest information from Web Search

…to produce the highest-accuracy answers.

✅ 4. Conversational Memory

The chatbot remembers previous messages.

Produces context-aware multi-turn conversations.

✅ 5. Groq LLM Integration

Uses Groq’s ultra-fast inference models like:

openai/gpt-oss-120b

meta-llama/llama-4-scout-17b-16e-instruct

qwen/qwen3-32b

✅ 6. Streamlit Frontend

A clean UI that supports:

Uploading files

Setting API keys

Chat interaction

Toggling RAG / Search / Hybrid modes

🛠️ Tech Stack
Component	Technology
LLM	Groq (ChatGroq)
Embeddings	HuggingFace (local)
Vector DB	Chroma
Memory	ConversationBufferMemory
Search Tools	Arxiv / Wikipedia / DuckDuckGo
UI	Streamlit
Document Parsing	PyPDFLoader / Docx2txtLoader / TextLoader
📦 Installation
1️⃣ Clone the repository
git clone https://github.com/yourusername/asknova.git
cd asknova

2️⃣ Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # (Linux/Mac)
venv\Scripts\activate     # (Windows)

3️⃣ Install dependencies
pip install -r requirements.txt

🔑 API Keys Needed
Groq API Key (mandatory for LLM)

Get one free at:
https://console.groq.com

Enter it in the Streamlit sidebar.

HuggingFace Embeddings (no API key required)

Embeddings run locally → completely free.

🚀 Run the App
streamlit run app.py


Your app will open at:

http://localhost:8501

📁 Project Structure
ASKNOVA/
│
├── app.py               # Main Streamlit application
├── chroma_store/        # Vector store directory
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── assets/              # (Optional) images, logos, etc.

🔥 How It Works (Workflow)
1. Upload Documents

PDF, TXT, DOCX accepted

Documents are split into chunks

Chunks are embedded using HuggingFace MiniLM

Stored in Chroma vector DB

2. Ask a Question

Choose mode:

✔ RAG Only

→ Uses only your documents

✔ Web Search Only

→ Queries Arxiv, Wikipedia, DuckDuckGo

✔ Hybrid (Recommended)

→ Combines both RAG + Web Search

3. Groq LLM Generates Answer

Fast inference

Cleanly formatted

Supports citations & hybrid reasoning