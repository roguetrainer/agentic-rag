"""
SmolAgents RAG Example
=======================
Demonstrates RAG with SmolAgents - agents write code to perform retrieval.
~60 lines of core logic.
"""

import os
from pathlib import Path

from smolagents import CodeAgent, tool, LiteLLMModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# Global variable to hold our vector store (tools need access)
_vectorstore = None


@tool
def search_document(query: str) -> str:
    """
    Search the loaded document for information relevant to the query.
    
    Args:
        query: The search query to find relevant information.
        
    Returns:
        Relevant text passages from the document.
    """
    global _vectorstore
    
    if _vectorstore is None:
        return "Error: No document has been loaded yet."
    
    # Perform similarity search
    docs = _vectorstore.similarity_search(query, k=3)
    
    # Format the results
    results = []
    for i, doc in enumerate(docs, 1):
        results.append(f"Passage {i}:\n{doc.page_content}")
    
    return "\n\n".join(results)


def load_document(document_path: str):
    """
    Load and index a PDF document for RAG.
    
    SmolAgents doesn't have built-in RAG, so we use LangChain
    components to create a searchable index, then expose it as a tool.
    """
    global _vectorstore
    
    # Load the PDF
    loader = PyPDFLoader(document_path)
    documents = loader.load()
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    _vectorstore = FAISS.from_documents(chunks, embeddings)


def setup_rag(document_path: str) -> CodeAgent:
    """
    Set up the SmolAgents RAG system.
    
    Creates a CodeAgent that can use the search_document tool
    to retrieve information and answer questions.
    """
    # Load the document into the vector store
    load_document(document_path)
    
    # Create the LLM model
    model = LiteLLMModel(model_id="gpt-4o-mini")
    
    # Create the agent with the search tool
    agent = CodeAgent(
        tools=[search_document],
        model=model,
        additional_authorized_imports=["json", "re"]
    )
    
    return agent


def query_rag(agent: CodeAgent, question: str) -> str:
    """
    Query the RAG system with a question.
    
    The agent will write and execute Python code to search
    the document and formulate an answer.
    """
    # Construct the prompt for the agent
    prompt = f"""Answer the following question based on the document content.
Use the search_document tool to find relevant information.

Question: {question}

Search the document for relevant information, then provide a clear answer based on what you find."""
    
    result = agent.run(prompt)
    return str(result)


def main():
    # Path to the sample document
    document_path = Path(__file__).parent.parent.parent / "data" / "sample.pdf"
    
    if not document_path.exists():
        print(f"Error: Document not found at {document_path}")
        print("Please add a PDF document to the data/ directory.")
        return
    
    print("SmolAgents RAG Example")
    print("=" * 50)
    
    # Setup the RAG system
    print("\n1. Loading and indexing document...")
    agent = setup_rag(str(document_path))
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
        answer = query_rag(agent, question)
        print(f"   A{i}: {answer}")
    
    print("\n" + "=" * 50)
    print("SmolAgents RAG Example Complete")


if __name__ == "__main__":
    main()
