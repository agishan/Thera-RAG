#!/usr/bin/env python3
"""
Question Testing Script for RAG System

Usage:
    python test_questions.py questions.txt --output results.json
    python test_questions.py questions.txt --k-values 5,10,15 --output comparison.json
"""

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Import our RAG components
import sys
sys.path.append("src")
from config import get_config
from rag_service import RAGService

class QuestionTester:
    """Test a list of questions against the RAG system"""

    def __init__(self, config_overrides: Dict = None):
        self.config = get_config()
        if config_overrides:
            self.config.update(config_overrides)
        self.rag_service = RAGService(self.config)

    def test_single_question(self, question: str, retrieval_k: int = None) -> Dict[str, Any]:
        """Test a single question and return detailed results"""
        start_time = time.time()

        try:
            result = self.rag_service.get_response(
                question=question,
                chat_history=[],
                retrieval_k=retrieval_k
            )

            elapsed = time.time() - start_time

            return {
                "question": question,
                "answer": result["answer"],
                "elapsed_time": elapsed,
                "retrieval_k": retrieval_k or self.config['retrieval_k'],
                "num_sources": len(result.get("source_documents", [])),
                "success": True,
                "error": None,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "question": question,
                "answer": None,
                "elapsed_time": elapsed,
                "retrieval_k": retrieval_k or self.config['retrieval_k'],
                "num_sources": 0,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def test_question_list(self, questions: List[str], retrieval_k_values: List[int] = None) -> Dict[str, Any]:
        """Test a list of questions with different K values"""
        if retrieval_k_values is None:
            retrieval_k_values = [self.config['retrieval_k']]

        results = {
            "metadata": {
                "total_questions": len(questions),
                "k_values_tested": retrieval_k_values,
                "config": {
                    "llm_model": self.config['llm_model'],
                    "embedding_model": self.config['embedding_model'],
                    "pinecone_index": self.config['pinecone_index_name'],
                    "pinecone_namespace": self.config['pinecone_namespace']
                },
                "start_time": datetime.now().isoformat()
            },
            "results": []
        }

        for i, question in enumerate(questions, 1):
            print(f"Testing question {i}/{len(questions)}: {question[:60]}...")

            question_results = []
            for k in retrieval_k_values:
                print(f"  Testing with K={k}")
                result = self.test_single_question(question, retrieval_k=k)
                question_results.append(result)

                # Small delay to avoid rate limiting
                time.sleep(1)

            results["results"].append({
                "question_index": i,
                "question": question,
                "k_value_results": question_results
            })

        results["metadata"]["end_time"] = datetime.now().isoformat()
        return results

def load_questions(file_path: str) -> List[str]:
    """Load questions from a text file (one per line)"""
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                questions.append(line)
    return questions

def main():
    parser = argparse.ArgumentParser(description="Test RAG system with a list of questions")
    parser.add_argument("questions_file", help="Text file with questions (one per line)")
    parser.add_argument("--output", "-o", default="test_results.json", help="Output JSON file")
    parser.add_argument("--k-values", "-k", default="15", help="Comma-separated K values to test (e.g., 5,10,15)")
    parser.add_argument("--config", help="JSON file with config overrides")

    args = parser.parse_args()

    # Parse K values
    k_values = [int(k.strip()) for k in args.k_values.split(',')]

    # Load config overrides if provided
    config_overrides = {}
    if args.config:
        with open(args.config, 'r') as f:
            config_overrides = json.load(f)

    # Load questions
    questions = load_questions(args.questions_file)
    print(f"Loaded {len(questions)} questions from {args.questions_file}")
    print(f"Testing with K values: {k_values}")

    # Run tests
    tester = QuestionTester(config_overrides)
    results = tester.test_question_list(questions, k_values)

    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {args.output}")

    # Print summary
    total_tests = len(questions) * len(k_values)
    successful_tests = sum(1 for q in results["results"] for r in q["k_value_results"] if r["success"])

    print(f"\nSummary:")
    print(f"  Total tests: {total_tests}")
    print(f"  Successful: {successful_tests}")
    print(f"  Failed: {total_tests - successful_tests}")

    if successful_tests > 0:
        avg_time = sum(r["elapsed_time"] for q in results["results"] for r in q["k_value_results"] if r["success"]) / successful_tests
        print(f"  Average response time: {avg_time:.2f}s")

if __name__ == "__main__":
    main()