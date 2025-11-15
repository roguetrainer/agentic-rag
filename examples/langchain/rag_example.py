"""
LangChain RAG Example
=====================
Demonstrates RAG with LangChain - more explicit control over each component.
~40 lines of core logic.
"""

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def setup_rag(document_path: str):
    """
    Load a document and create a retrieval QA chain.
    
    LangChain requires explicit configuration of each component,
    offering more control but requiring more code.
    """
    # Load the PDF document
    loader = PyPDFLoader(document_path)
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Create the retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # Retrieve top 3 chunks
    )
    
    # Configure the LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create a custom prompt template
    prompt_template = """Use the following pieces of context to answer the question. 
If you don't know the answer based on the context, say so.

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain


def query_rag(qa_chain, question: str) -> str:
    """
    Query the RAG system with a question.
    
    Returns the generated answer based on retrieved context.
    """
    result = qa_chain.invoke({"query": question})
    return result["result"]


def main():
    # Path to the sample document
    document_path = Path(__file__).parent.parent.parent / "data" / "sample.pdf"
    
    if not document_path.exists():
        print(f"Error: Document not found at {document_path}")
        print("Please add a PDF document to the data/ directory.")
        return
    
    print("LangChain RAG Example")
    print("=" * 50)
    
    # Setup the RAG system
    print("\n1. Loading and indexing document...")
    qa_chain = setup_rag(str(document_path))
    print("   ✓ Document indexed successfully")
    
    # Example questions
    questions = [
        "What is the main topic of this document?",
        "What are the key findings or conclusions?",
        "Are there any limitations mentioned?"
    ]
    
    # Query the system
    print("\n2. Querying the RAG system:")
    for i, question in enumerate(questions, 1):
        print(f"\n   Q{i}: {question}")
        answer = query_rag(qa_chain, question)
        print(f"   A{i}: {answer}")
    
    print("\n" + "=" * 50)
    print("LangChain RAG Example Complete")


if __name__ == "__main__":
    main()
