"""
LlamaIndex RAG Example
=======================
Demonstrates RAG with LlamaIndex - the most streamlined approach.
~25 lines of core logic.
"""

import os
from pathlib import Path

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


def setup_rag(document_path: str) -> VectorStoreIndex:
    """
    Load a document and create a queryable index.
    
    LlamaIndex handles chunking, embedding, and indexing automatically.
    """
    # Configure the LLM and embedding model
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    
    # Load documents from the specified path
    documents = SimpleDirectoryReader(
        input_files=[document_path]
    ).load_data()
    
    # Create the index - this chunks, embeds, and stores automatically
    index = VectorStoreIndex.from_documents(documents)
    
    return index


def query_rag(index: VectorStoreIndex, question: str) -> str:
    """
    Query the RAG system with a question.
    
    Returns the generated answer based on retrieved context.
    """
    query_engine = index.as_query_engine(
        similarity_top_k=3,  # Retrieve top 3 relevant chunks
        response_mode="compact"  # Concise responses
    )
    
    response = query_engine.query(question)
    return str(response)


def main():
    # Path to the sample document
    document_path = Path(__file__).parent.parent.parent / "data" / "sample.pdf"
    
    if not document_path.exists():
        print(f"Error: Document not found at {document_path}")
        print("Please add a PDF document to the data/ directory.")
        return
    
    print("LlamaIndex RAG Example")
    print("=" * 50)
    
    # Setup the RAG system
    print("\n1. Loading and indexing document...")
    index = setup_rag(str(document_path))
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
        answer = query_rag(index, question)
        print(f"   A{i}: {answer}")
    
    print("\n" + "=" * 50)
    print("LlamaIndex RAG Example Complete")


if __name__ == "__main__":
    main()
