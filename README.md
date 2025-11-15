# Agentic RAG Framework Comparison

A side-by-side comparison of lightweight agentic AI frameworks for Retrieval-Augmented Generation (RAG). This repository implements the same RAG task across three popular frameworks to help you understand their trade-offs.

![Agentic RAG](Agentic-RAG.png)

## Frameworks Compared

- **LlamaIndex** - Purpose-built for RAG, minimal boilerplate
- **LangChain** - General-purpose LLM orchestration with strong RAG support
- **SmolAgents** - Minimalist, code-first approach from Hugging Face

## The Task

Each implementation performs the same task: **Q&A over a PDF document**. We use a sample technical document and answer identical questions across all three frameworks, comparing:

- Lines of code required
- Setup complexity
- Query accuracy
- Flexibility and extensibility

## Related Repository

For multi-agent orchestration patterns (using AutoGen and CrewAI), see our companion repository:
**[multi-agent-team](https://github.com/roguetrainer/multi-agent-team)**

## Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key (or other LLM provider)

### Installation

```bash
# Clone the repository
git clone https://github.com/roguetrainer/agentic-rag.git
cd agentic-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies for all frameworks
pip install -r requirements.txt

# Set your API key
export OPENAI_API_KEY="your-key-here"
```

### Running the Examples

Each framework has its own directory with a standalone implementation:

```bash
# LlamaIndex (simplest for RAG)
python examples/llamaindex/rag_example.py

# LangChain (most flexible)
python examples/langchain/rag_example.py

# SmolAgents (code-first approach)
python examples/smolagents/rag_example.py
```

### Running the Comparison

To run all three frameworks against the same questions and compare results:

```bash
python compare_frameworks.py
```

## Project Structure

```
agentic-rag/
├── README.md
├── requirements.txt
├── compare_frameworks.py          # Runs all frameworks and compares
├── data/
│   └── sample.pdf                 # Test document
├── examples/
│   ├── llamaindex/
│   │   └── rag_example.py
│   ├── langchain/
│   │   └── rag_example.py
│   └── smolagents/
│       └── rag_example.py
└── docs/
    ├── COMPARISON.md              # Detailed comparison analysis
    └── SETUP.md                   # Extended setup instructions
```

## Sample Results

| Metric | LlamaIndex | LangChain | SmolAgents |
|--------|-----------|-----------|------------|
| Lines of Code | ~25 | ~40 | ~60 |
| Setup Complexity | Low | Medium | Medium |
| RAG-Specific Features | Excellent | Good | Basic |
| Extensibility | Good | Excellent | Good |
| Learning Curve | Low | Medium | Low |

*Results based on implementing identical functionality. See [docs/COMPARISON.md](docs/COMPARISON.md) for detailed analysis.*

## Key Findings

**LlamaIndex** excels when RAG is your primary use case. The framework was built specifically for connecting LLMs to data, and it shows - you get sophisticated chunking, indexing, and retrieval with minimal code.

**LangChain** offers the most flexibility. While slightly more verbose for simple RAG, it provides extensive options for customizing every component and integrates well if you plan to add more complex agent behaviors later.

**SmolAgents** takes a different philosophical approach. Rather than abstracting away the retrieval process, it lets your agent write code to perform retrieval, which can be more transparent but requires more setup for basic RAG.

## Customization

### Using Different Documents

Place your PDF in the `data/` directory and update the file path in each example:

```python
# In any example file
document_path = "data/your_document.pdf"
```

### Using Different LLM Providers

Each framework supports multiple providers. See the individual example files for configuration options, or check [docs/SETUP.md](docs/SETUP.md) for provider-specific instructions.

### Adding Custom Questions

Edit the `questions` list in `compare_frameworks.py`:

```python
questions = [
    "Your first question here?",
    "Your second question here?",
    # Add more as needed
]
```

## Contributing

Contributions welcome! Areas of interest:

- Adding more frameworks to the comparison
- Improving example implementations
- Adding benchmarking metrics
- Documentation improvements

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [LlamaIndex](https://www.llamaindex.ai/)
- [LangChain](https://www.langchain.com/)
- [SmolAgents](https://huggingface.co/docs/smolagents)
