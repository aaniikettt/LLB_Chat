import streamlit as st
import altair as alt

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Jupyter-specific imports
from IPython.display import display, Markdown

# Set environment variable for protobuf
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


def load_documents():
    local_path1 = "indian_constituition_2024_updated.pdf"
    local_path2 = "BNSS.pdf"
    if local_path1:
        loader1 = PyMuPDFLoader(file_path=local_path1)
        loader2 = PyMuPDFLoader(file_path=local_path2)
        data1 = loader1.load()
        data2 = loader2.load()
        print(f"PDF files loaded successfully: {local_path}")
    else:
        print("Database Missing!!!")

    return data1, data2


def text_chunks(data1, data2):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks1 = text_splitter.split_documents(data1)
    chunks2 = text_splitter.split_documents(data2)
    print(f"Data 1 Text split into {len(chunks1)} chunks")
    print(f"data 2 Text split into {len(chunks2)} chunks")
    return chunks1, chunks2


def vec_db(chunks1, chunks2):
    vector_db1 = Chroma.from_documents(documents=chunks1, embedding=OllamaEmbeddings(model="llama3.2"), collection_name="local-rag")
    vector_db2 = Chroma.from_documents(documents=chunks2, embedding=OllamaEmbeddings(model="llama3.2"), collection_name="local-rag")
    print("Vector database created successfully")
    return vector_db1, vector_db2



def prompt_definition(vector_db1, vector_db2):
    # Set up LLM and retrieval
    local_model = "llama3.2" 
    llm = ChatOllama(model=local_model)
    QUERY_PROMPT = PromptTemplate(
                    input_variables=["question"],
                    template="""You are an AI language model assistant. Your task is to generate 2
                    different versions of the given user question to retrieve relevant documents from
                    a vector database. By generating multiple perspectives on the user question, your
                    goal is to help the user overcome some of the limitations of the distance-based
                    similarity search. Provide these alternative questions separated by newlines.
                    Original question: {question}""",
                )

    # Set up retriever
    retriever1 = MultiQueryRetriever.from_llm(
                vector_db1.as_retriever(), 
                llm,
                prompt=QUERY_PROMPT)
    
    retriever2 = MultiQueryRetriever.from_llm(
                vector_db2.as_retriever(), 
                llm,
                prompt=QUERY_PROMPT)

    # RAG prompt template
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    # Create chain
    chain1 = (
        {"context": retriever1, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    chain2 = (
        {"context": retriever2, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain1, chain2


def chat_with_pdf(question, chain1, chain2):
    print(f"How can I help you today?/n")
    """
    Chat with the PDF using the RAG chain.
    """
    d1 = display(Markdown(chain1.invoke(question)))
    d2 = display(Markdown(chain2.invoke(question)))

    return d1, d2


def process_start():
    data1, data2 = load_documents()
    t_chunks1, t_chunks2 = text_chunks(data1, data2)
    create_db1, create_db2 = vec_db(t_chunks1, t_chunks2)
    prompt_llm1, prompt_llm2 = prompt_definition(create_db1, create_db2)
    




