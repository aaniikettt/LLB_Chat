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



local_path = "indian_rules.pdf"

if local_path:
    loader = PyMuPDFLoader(file_path=local_path)
    data = loader.load()
    print(f"PDF loaded successfully: {local_path}")
else:
    print("Upload a PDF file")


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(data)
print(f"Text split into {len(chunks)} chunks")

# Create vector database
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="llama3.2"),
    collection_name="local-rag"
)
print("Vector database created successfully")

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
retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), 
    llm,
    prompt=QUERY_PROMPT
)

# RAG prompt template
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)


# Create chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def chat_with_pdf(question):
    """
    Chat with the PDF using the RAG chain.
    """
    return display(Markdown(chain.invoke(question)))


def use_llm_llb():
    print(f"what can I help you with?")
    query = input()
    answer = chat_with_pdf(query)

    
    return answer

