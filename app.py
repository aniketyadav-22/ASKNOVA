import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma as ChromaDB
from langchain_core.embeddings import Embeddings
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_classic.agents import initialize_agent, AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler
from langchain_community.embeddings import HuggingFaceEmbeddings

import tempfile

#streamlit ui
st.set_page_config(page_title="ASK NOVA",layout="wide")
st.title("ASK NOVA : just ask anything")

#GROQ api key
groq_api_key = st.sidebar.text_input("Enter YOUR GROQ API Key", type="password")


#model selection
model_name=st.sidebar.selectbox(
    'Choose Groq Model',
    ["openai/gpt-oss-120b","meta-llama/llama-4-scout-17b-16e-instruct","qwen/qwen3-32b"]
)

if groq_api_key:
    os.environ["GROQ_API_KEY"]=groq_api_key
else :
    st.warning("Please enter your GROQ API KEY")
    st.stop()

#LLM
llm =ChatGroq(
    groq_api_key=groq_api_key,
    model_name=model_name,
    temperature=0.3,
    streaming= True

)

#Embedding using groq
import google.generativeai as genai
from langchain_core.embeddings import Embeddings




embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#vector stoe(chroma)

dir= "./chroma_store"
db=None

#TEXT SPILTTER

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=150
)
   
#file uploader
st.subheader("Upload files for analysis and knowledge")
uploaded_files=st.file_uploader(
    "Upload PDF / TXT / DOCX files",
    type=["pdf","txt","docx"],
    accept_multiple_files=True
)

if st.button("Process & Embed Files"):
    all_docs=[]
    for file in uploaded_files:
        temp_path = os.path.join(tempfile.gettempdir(),file.name)
        with open(temp_path,"wb") as f:
            f.write(file.read())
        if file.name.endswith(".pdf"):
            loader= PyPDFLoader(temp_path)
        elif file.name.endswith(".txt"):
            loader= TextLoader(temp_path)
        else:
            loader= Docx2txtLoader(temp_path)

        docs= loader.load()    
        split_docs = text_splitter.split_documents(docs)
        all_docs.extend(split_docs)
    
    st.info("Creating embeddings for {len(all_docs)} chunks...")

    db = Chroma.from_documents(
        documents=all_docs,
        embedding=embedding_model,
        persist_directory=dir
    )
    st.success("Files embedded and added to vector database")

#load existing db if available
if os.path.exists(dir):
    try:
        db= ChromaDB(
            persist_directory=dir,
            embedding_function= embedding_model
        )
    except:
        pass

#chat memory
memory= ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

#Retriever + Rag chain
retriever = None
rag_chain = None

if db:
    retriever=db.as_retriever(search_top_k=4)
    rag_chain= ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )


#Arxiv and wikipedia
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)


api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

search = DuckDuckGoSearchRun(name="Search")
tools = [arxiv,wiki,search]

#Agent for external research + reasoning
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors= True
)

#chat ui
st.markdown("-----")
st.header("Chat with AskNOVA")

query = st.text_area("Ask anything!",height=120)

col1,col2= st.columns(2)
use_rag = col1.checkbox("Use RAG (Vector DB)")
use_web =col2.checkbox("Use Web Search Tools")

if st.button("Send"):
    if not query.strip():
        st.error("Enter a message")
        st.stop()
    callback =  StreamlitCallbackHandler(st.container())

    if use_web and not use_rag:
        st.write("### 🌍 Web Search + Agent Tools Running...")
        response = agent.run(query, callbacks=[callback])
        st.success("Done!")
        st.write(response)
    
    elif use_rag and not use_web and rag_chain:
        st.write("### 📚 RAG Retrieval Running...")
        result = rag_chain({"question": query}, callbacks=[callback])
        st.success("Done!")
        st.write(result["answer"])
    elif use_rag and use_web and rag_chain:
        st.write("### 🔥 Hybrid Mode: RAG + Web Search Running...")

        # Step 1: RAG answer
        rag_result = rag_chain({"question": query})

        rag_context = rag_result["answer"]

        # Step 2: Web search answer
        web_result = agent.run(query)

        # Step 3: Combine both
        final_prompt = f"""
You are AskNOVA. Use BOTH RAG knowledge and Web Search results to produce the best possible answer.

### RAG Knowledge:
{rag_context}

### Web Search Results:
{web_result}

### User Question:
{query}

Now combine both sources, remove duplicates, resolve contradictions, and produce the best possible final answer.
"""

        st.write("### 🤖 Combining RAG + Web Search...")
        final_response = llm.invoke(final_prompt)

        st.success("Done!")
        st.write(final_response.content)

    else:
        st.write("### 🤖 LLM Answer (No RAG, No Tools)")
        resp = llm.invoke(query)
        st.write(resp.content)






