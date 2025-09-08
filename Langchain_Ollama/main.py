# ------------------ main.py ------------------
# Import required libraries from LangChain and Ollama
from langchain_community.document_loaders import PyPDFLoader  # To load PDFs
from langchain_text_splitters import RecursiveCharacterTextSplitter  # To split large text into smaller chunks
from langchain_community.vectorstores import FAISS  # For storing document embeddings for search
from langchain_core.vectorstores import InMemoryVectorStore  # Optional: in-memory storage
from langchain_ollama import OllamaEmbeddings  # To generate embeddings using Ollama model
from langchain_core.prompts import ChatPromptTemplate  # To create prompt templates for the model
from langchain_ollama.llms import OllamaLLM  # Ollama language model
import os  # For working with file paths and directories

# Directory where uploaded PDFs will be stored
pdfs_directory = 'pdfs/'

# Initialize Ollama embeddings (for converting text into vectors)
embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")

# Initialize Ollama LLM (for generating answers)
model = OllamaLLM(model="deepseek-r1:1.5b")

# Template for how the AI should answer questions
template = """
You are an assistant that answers questions. Using the following retrieved information, answer the user question. 
If you don't know the answer, say that you don't know. Use up to three sentences, keeping the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""


# Function to save uploaded PDF to the 'pdfs/' folder
def upload_pdf(file):
    # Create 'pdfs/' folder if it doesn't exist
    if not os.path.exists(pdfs_directory):
        os.makedirs(pdfs_directory)
    # Save uploaded file to the folder
    with open(os.path.join(pdfs_directory, file.name), "wb") as f:
        f.write(file.getbuffer())


# Function to create a vector store from a PDF
def create_vector_store(file_path):
    # Load the PDF as documents
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split large documents into smaller chunks for better processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # Max characters per chunk
        chunk_overlap=300,  # Overlap between chunks
        add_start_index=True
    )

    # Split the documents
    chunked_docs = text_splitter.split_documents(documents)

    # Create a FAISS vector store from the chunked documents
    db = FAISS.from_documents(chunked_docs, embeddings)
    return db


# Function to retrieve top relevant documents for a query
def retrieve_docs(db, query, k=4):
    # Print retrieved docs in console (optional)
    print(db.similarity_search(query))
    # Return top k documents relevant to the query
    return db.similarity_search(query, k)


# Function to ask a question to the PDF using the AI
def question_pdf(question, documents):
    # Combine content of retrieved documents as context
    context = "\n\n".join([doc.page_content for doc in documents])
    # Create a prompt from the template
    prompt = ChatPromptTemplate.from_template(template)
    # Connect the prompt to the AI model
    chain = prompt | model
    # Invoke the model with question + context and return answer
    return chain.invoke({"question": question, "context": context})