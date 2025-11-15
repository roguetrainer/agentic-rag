# Extended Setup Guide

This guide covers advanced setup options including alternative LLM providers, vector stores, and environment configuration.

## Prerequisites

- Python 3.10 or higher
- pip (Python package installer)
- Git
- An API key for your chosen LLM provider

## Basic Setup

```bash
# Clone the repository
git clone https://github.com/roguetrainer/agentic-rag.git
cd agentic-rag

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install all dependencies
pip install -r requirements.txt
```

## Environment Configuration

Create a `.env` file in the project root:

```bash
# OpenAI (default)
OPENAI_API_KEY=your-openai-key-here

# Alternative providers
ANTHROPIC_API_KEY=your-anthropic-key-here
COHERE_API_KEY=your-cohere-key-here

# Optional: Custom model endpoints
OPENAI_API_BASE=https://your-custom-endpoint.com/v1
```

The examples will automatically load these from your environment or `.env` file.

## Using Alternative LLM Providers

### Anthropic Claude

**LlamaIndex**:
```python
from llama_index.llms.anthropic import Anthropic
Settings.llm = Anthropic(model="claude-3-sonnet-20240229")
```

**LangChain**:
```python
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-sonnet-20240229")
```

**SmolAgents**:
```python
model = LiteLLMModel(model_id="anthropic/claude-3-sonnet-20240229")
```

### Local Models (Ollama)

First, install and run Ollama:
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3
```

**LlamaIndex**:
```python
from llama_index.llms.ollama import Ollama
Settings.llm = Ollama(model="llama3", request_timeout=120.0)
```

**LangChain**:
```python
from langchain_community.llms import Ollama
llm = Ollama(model="llama3")
```

**SmolAgents**:
```python
model = LiteLLMModel(model_id="ollama/llama3")
```

### Cohere

```bash
pip install cohere
```

**LlamaIndex**:
```python
from llama_index.llms.cohere import Cohere
Settings.llm = Cohere(model="command-r-plus")
```

**LangChain**:
```python
from langchain_cohere import ChatCohere
llm = ChatCohere(model="command-r-plus")
```

## Alternative Embedding Models

### Cohere Embeddings

```python
# LlamaIndex
from llama_index.embeddings.cohere import CohereEmbedding
Settings.embed_model = CohereEmbedding(model_name="embed-english-v3.0")

# LangChain
from langchain_cohere import CohereEmbeddings
embeddings = CohereEmbeddings(model="embed-english-v3.0")
```

### Local Embeddings (Hugging Face)

```python
# LlamaIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# LangChain
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
```

## Alternative Vector Stores

### Chroma (Persistent)

```bash
pip install chromadb
```

```python
# LlamaIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("documents")
vector_store = ChromaVectorStore(chroma_collection=collection)

# LangChain
from langchain_community.vectorstores import Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

### Qdrant

```bash
pip install qdrant-client
```

```python
# LlamaIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

client = QdrantClient(path="./qdrant_db")
vector_store = QdrantVectorStore(client=client, collection_name="documents")

# LangChain
from langchain_community.vectorstores import Qdrant
vectorstore = Qdrant.from_documents(
    chunks,
    embeddings,
    path="./qdrant_db",
    collection_name="documents"
)
```

### Pinecone (Cloud)

```bash
pip install pinecone-client
```

```python
import pinecone
pinecone.init(api_key="your-key", environment="your-env")

# LlamaIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
vector_store = PineconeVectorStore(pinecone_index=pinecone.Index("documents"))

# LangChain
from langchain_community.vectorstores import Pinecone
vectorstore = Pinecone.from_documents(chunks, embeddings, index_name="documents")
```

## Performance Tuning

### Chunking Strategies

**LlamaIndex** - Sentence Window:
```python
from llama_index.core.node_parser import SentenceWindowNodeParser

node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)
```

**LangChain** - Semantic Chunking:
```python
from langchain_experimental.text_splitter import SemanticChunker

text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile"
)
```

### Retrieval Optimization

**Hybrid Search** (combining keyword and semantic):
```python
# LlamaIndex
from llama_index.core.retrievers import QueryFusionRetriever

retriever = QueryFusionRetriever(
    [vector_retriever, keyword_retriever],
    similarity_top_k=5,
    num_queries=3
)

# LangChain
from langchain.retrievers import EnsembleRetriever

retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)
```

**Reranking**:
```python
# LlamaIndex
from llama_index.postprocessor.cohere_rerank import CohereRerank

reranker = CohereRerank(top_n=3)
query_engine = index.as_query_engine(
    node_postprocessors=[reranker]
)

# LangChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

compressor = CohereRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

## Troubleshooting

### Common Issues

**ModuleNotFoundError**:
```bash
# Ensure you're in the virtual environment
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

**API Key Not Found**:
```bash
# Check your environment variable
echo $OPENAI_API_KEY

# Or use python-dotenv
pip install python-dotenv
```

**PDF Loading Issues**:
```bash
# Install additional PDF dependencies
pip install pymupdf  # Alternative PDF loader
```

**Memory Issues with Large Documents**:
```python
# Process documents in batches
from llama_index.core import Settings
Settings.chunk_size = 512  # Smaller chunks
Settings.chunk_overlap = 50
```

### Getting Help

- Check the framework-specific documentation
- Review the example code comments
- Open an issue in this repository
- Consult the companion [multi-agent-team](https://github.com/roguetrainer/multi-agent-team) repository

## Next Steps

1. Add a PDF document to the `data/` directory
2. Run the basic examples to ensure setup is correct
3. Try the comparison script to see frameworks side-by-side
4. Experiment with alternative providers and configurations
5. Explore the multi-agent patterns in the companion repository
