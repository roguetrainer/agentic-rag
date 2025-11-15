"""
Framework Comparison Script
============================
Runs the same RAG task across all three frameworks and compares results.
"""

import time
import sys
from pathlib import Path

# Add examples to path
sys.path.insert(0, str(Path(__file__).parent))

from examples.llamaindex.rag_example import setup_rag as setup_llamaindex, query_rag as query_llamaindex
from examples.langchain.rag_example import setup_rag as setup_langchain, query_rag as query_langchain
from examples.smolagents.rag_example import setup_rag as setup_smolagents, query_rag as query_smolagents


def run_comparison():
    """Run all frameworks against the same questions and compare."""
    
    document_path = Path(__file__).parent / "data" / "sample.pdf"
    
    if not document_path.exists():
        print("=" * 70)
        print("ERROR: No sample document found!")
        print("=" * 70)
        print(f"\nPlease add a PDF document to: {document_path}")
        print("\nSuggested test documents:")
        print("  - A technical paper or research article")
        print("  - A company report or whitepaper")
        print("  - Any 10-20 page PDF with substantive content")
        print("\nThe same document will be used across all three frameworks")
        print("to ensure a fair comparison.")
        return
    
    # Questions to test across all frameworks
    questions = [
        "What is the main topic of this document?",
        "What are the key findings or conclusions?",
        "Are there any limitations or challenges mentioned?",
    ]
    
    print("=" * 70)
    print("AGENTIC RAG FRAMEWORK COMPARISON")
    print("=" * 70)
    print(f"\nDocument: {document_path.name}")
    print(f"Questions: {len(questions)}")
    print("\n" + "-" * 70)
    
    results = {}
    
    # Test LlamaIndex
    print("\n[1/3] LLAMAINDEX")
    print("-" * 70)
    try:
        start = time.time()
        index = setup_llamaindex(str(document_path))
        setup_time = time.time() - start
        print(f"Setup time: {setup_time:.2f}s")
        
        results["llamaindex"] = {
            "setup_time": setup_time,
            "answers": [],
            "query_times": []
        }
        
        for q in questions:
            start = time.time()
            answer = query_llamaindex(index, q)
            query_time = time.time() - start
            results["llamaindex"]["answers"].append(answer)
            results["llamaindex"]["query_times"].append(query_time)
            print(f"\nQ: {q}")
            print(f"A: {answer[:500]}..." if len(answer) > 500 else f"A: {answer}")
            print(f"Time: {query_time:.2f}s")
    except Exception as e:
        print(f"Error: {e}")
        results["llamaindex"] = {"error": str(e)}
    
    # Test LangChain
    print("\n\n[2/3] LANGCHAIN")
    print("-" * 70)
    try:
        start = time.time()
        qa_chain = setup_langchain(str(document_path))
        setup_time = time.time() - start
        print(f"Setup time: {setup_time:.2f}s")
        
        results["langchain"] = {
            "setup_time": setup_time,
            "answers": [],
            "query_times": []
        }
        
        for q in questions:
            start = time.time()
            answer = query_langchain(qa_chain, q)
            query_time = time.time() - start
            results["langchain"]["answers"].append(answer)
            results["langchain"]["query_times"].append(query_time)
            print(f"\nQ: {q}")
            print(f"A: {answer[:500]}..." if len(answer) > 500 else f"A: {answer}")
            print(f"Time: {query_time:.2f}s")
    except Exception as e:
        print(f"Error: {e}")
        results["langchain"] = {"error": str(e)}
    
    # Test SmolAgents
    print("\n\n[3/3] SMOLAGENTS")
    print("-" * 70)
    try:
        start = time.time()
        agent = setup_smolagents(str(document_path))
        setup_time = time.time() - start
        print(f"Setup time: {setup_time:.2f}s")
        
        results["smolagents"] = {
            "setup_time": setup_time,
            "answers": [],
            "query_times": []
        }
        
        for q in questions:
            start = time.time()
            answer = query_smolagents(agent, q)
            query_time = time.time() - start
            results["smolagents"]["answers"].append(answer)
            results["smolagents"]["query_times"].append(query_time)
            print(f"\nQ: {q}")
            print(f"A: {answer[:500]}..." if len(answer) > 500 else f"A: {answer}")
            print(f"Time: {query_time:.2f}s")
    except Exception as e:
        print(f"Error: {e}")
        results["smolagents"] = {"error": str(e)}
    
    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nSetup Times:")
    for framework, data in results.items():
        if "error" not in data:
            print(f"  {framework:12s}: {data['setup_time']:.2f}s")
        else:
            print(f"  {framework:12s}: ERROR - {data['error']}")
    
    print("\nAverage Query Times:")
    for framework, data in results.items():
        if "error" not in data:
            avg_time = sum(data["query_times"]) / len(data["query_times"])
            print(f"  {framework:12s}: {avg_time:.2f}s")
    
    print("\nCode Complexity (approximate lines of core logic):")
    print(f"  {'llamaindex':12s}: ~25 lines")
    print(f"  {'langchain':12s}: ~40 lines")
    print(f"  {'smolagents':12s}: ~60 lines")
    
    print("\n" + "=" * 70)
    print("For multi-agent orchestration examples, see:")
    print("https://github.com/roguetrainer/multi-agent-team")
    print("=" * 70)


if __name__ == "__main__":
    run_comparison()
