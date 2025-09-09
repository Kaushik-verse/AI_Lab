import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

pdfs_directory = 'pdfs/'

embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
model = OllamaLLM(model="deepseek-r1:1.5b")

template = """
You are an assistant that answers questions. Using the following retrieved information, answer the user question. If you don't know the answer, say that you don't know. Use up to three sentences, keeping the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

def upload_pdf(file):
    if not os.path.exists(pdfs_directory):
        os.makedirs(pdfs_directory)
    with open(os.path.join(pdfs_directory, file.name), "wb") as f:
        f.write(file.getbuffer())

def create_vector_store(file_paths):
    all_documents = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        all_documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        add_start_index=True
    )
    chunked_docs = text_splitter.split_documents(all_documents)

    db = FAISS.from_documents(chunked_docs, embeddings)

    return db, chunked_docs

def retrieve_docs(db, query, k=4):
    return db.similarity_search(query, k)

def question_pdf(question, documents, context_with_history):
    context = context_with_history + "\n\n" + "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

def summarize_pdf(documents):
    summary_text = "\n\n".join([doc.page_content for doc in documents])
    # Simple summary logic for demo purposes
    return summary_text[:2000] + "..." if len(summary_text) > 2000 else summary_text