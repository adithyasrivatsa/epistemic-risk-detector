#!/usr/bin/env python3
"""
Sample script demonstrating the Epistemic Risk Detector.

This script shows how to:
1. Index a corpus of documents
2. Analyze an LLM response for epistemic risk
3. Render results in different formats
"""

import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import Config
from src.pipeline import EpistemicRiskDetector


def main():
    # Initialize with default config
    config = Config()
    detector = EpistemicRiskDetector(config)

    # Index the example corpus
    corpus_path = Path(__file__).parent.parent / "example_corpus"
    if corpus_path.exists():
        print(f"Indexing corpus from {corpus_path}...")
        chunk_count = detector.index_corpus(str(corpus_path))
        print(f"Indexed {chunk_count} chunks\n")
    else:
        print("Warning: Example corpus not found. Results will show 'no evidence'.\n")

    # Sample LLM responses to analyze
    test_responses = [
        # Grounded claim
        "Python was created by Guido van Rossum and first released in 1991.",
        
        # Hallucination - GIL was NOT removed in 3.12
        "Python 3.12 completely removed the Global Interpreter Lock (GIL), "
        "allowing true multi-threaded execution.",
        
        # Partially correct
        "The Transformer architecture was introduced in 2017 and GPT-3 has "
        "175 billion parameters. GPT-4 was released in 2022.",
        
        # Mixed claims
        "Python is named after Monty Python. Django was released in 2010. "
        "NumPy was created by Travis Oliphant in 2005.",
    ]

    for i, response in enumerate(test_responses, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: Analyzing response...")
        print(f"{'='*60}")
        print(f"Input: \"{response[:80]}{'...' if len(response) > 80 else ''}\"")
        print()

        # Analyze
        result = detector.analyze(response)

        # Render CLI output
        print(detector.render_cli(result))

        # Also show JSON structure for one example
        if i == 2:
            print("\n--- JSON Output (for API/web use) ---")
            print(detector.render_json(result)[:500] + "...")


if __name__ == "__main__":
    main()
