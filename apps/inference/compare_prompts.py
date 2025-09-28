#!/usr/bin/env python3
"""
Vanilla vs RAG Prompt Comparison Script

This script tests the same questions against:
1. Vanilla LLM (no RAG, just the LLM)
2. RAG-enhanced LLM (with document retrieval)

Usage:
    python compare_prompts.py questions.txt --output comparison.json
    python compare_prompts.py questions.txt --k-values 5,15 --output detailed_comparison.json
"""

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Import our components
import sys
sys.path.append("src")
from config import get_config
from rag_service import RAGService

# Import Google Generative AI directly for vanilla prompts
from langchain_google_genai import ChatGoogleGenerativeAI

class PromptComparator:
    """Compare vanilla LLM vs RAG-enhanced responses"""

    def __init__(self, config_overrides: Dict = None):
        self.config = get_config()
        if config_overrides:
            self.config.update(config_overrides)

        # RAG service
        self.rag_service = RAGService(self.config)

        # Vanilla LLM (no RAG)
        self.vanilla_llm = ChatGoogleGenerativeAI(
            model=self.config['llm_model'],
            temperature=self.config['llm_temperature'],
            max_tokens=self.config['llm_max_tokens'],
            google_api_key=self.config['google_api_key']
        )

        # Medical prompt for vanilla LLM
        self.vanilla_prompt = """You are a knowledgeable medical assistant with expertise in clinical guidelines and viscoelastic testing.

Please provide a concise, evidence-based answer to the following medical question. Be accurate and precise. If you're uncertain about specific details, please indicate that.

Question: {question}

Answer:"""

    def test_vanilla_response(self, question: str) -> Dict[str, Any]:
        """Test vanilla LLM response (no RAG)"""
        start_time = time.time()

        try:
            # Format the prompt
            formatted_prompt = self.vanilla_prompt.format(question=question)

            # Get response from vanilla LLM
            response = self.vanilla_llm.invoke(formatted_prompt)
            answer = response.content

            elapsed = time.time() - start_time

            return {
                "question": question,
                "answer": answer,
                "elapsed_time": elapsed,
                "method": "vanilla",
                "num_sources": 0,
                "retrieval_k": None,
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
                "method": "vanilla",
                "num_sources": 0,
                "retrieval_k": None,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def test_rag_response(self, question: str, retrieval_k: int = None) -> Dict[str, Any]:
        """Test RAG-enhanced response"""
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
                "method": "rag",
                "num_sources": len(result.get("source_documents", [])),
                "retrieval_k": retrieval_k or self.config['retrieval_k'],
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
                "method": "rag",
                "num_sources": 0,
                "retrieval_k": retrieval_k or self.config['retrieval_k'],
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def compare_question_list(self, questions: List[str], retrieval_k_values: List[int] = None) -> Dict[str, Any]:
        """Compare vanilla vs RAG for a list of questions"""
        if retrieval_k_values is None:
            retrieval_k_values = [self.config['retrieval_k']]

        results = {
            "metadata": {
                "total_questions": len(questions),
                "k_values_tested": retrieval_k_values,
                "comparison_methods": ["vanilla", "rag"],
                "config": {
                    "llm_model": self.config['llm_model'],
                    "embedding_model": self.config['embedding_model'],
                    "pinecone_index": self.config['pinecone_index_name'],
                    "pinecone_namespace": self.config['pinecone_namespace']
                },
                "start_time": datetime.now().isoformat()
            },
            "comparisons": []
        }

        for i, question in enumerate(questions, 1):
            print(f"\\nComparing question {i}/{len(questions)}: {question[:60]}...")

            comparison = {
                "question_index": i,
                "question": question,
                "responses": []
            }

            # Test vanilla response
            print("  Testing vanilla LLM...")
            vanilla_result = self.test_vanilla_response(question)
            comparison["responses"].append(vanilla_result)

            # Small delay to avoid rate limiting
            time.sleep(2)

            # Test RAG responses for each K value
            for k in retrieval_k_values:
                print(f"  Testing RAG with K={k}...")
                rag_result = self.test_rag_response(question, retrieval_k=k)
                comparison["responses"].append(rag_result)

                # Small delay to avoid rate limiting
                time.sleep(2)

            results["comparisons"].append(comparison)

        results["metadata"]["end_time"] = datetime.now().isoformat()
        return results

    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze comparison results and generate summary statistics"""
        analysis = {
            "summary": {},
            "performance_comparison": {},
            "success_rates": {}
        }

        vanilla_responses = []
        rag_responses = []

        for comparison in results["comparisons"]:
            for response in comparison["responses"]:
                if response["method"] == "vanilla":
                    vanilla_responses.append(response)
                else:  # rag
                    rag_responses.append(response)

        # Success rates
        analysis["success_rates"]["vanilla"] = sum(1 for r in vanilla_responses if r["success"]) / len(vanilla_responses) if vanilla_responses else 0
        analysis["success_rates"]["rag"] = sum(1 for r in rag_responses if r["success"]) / len(rag_responses) if rag_responses else 0

        # Performance comparison
        successful_vanilla = [r for r in vanilla_responses if r["success"]]
        successful_rag = [r for r in rag_responses if r["success"]]

        if successful_vanilla:
            analysis["performance_comparison"]["vanilla_avg_time"] = sum(r["elapsed_time"] for r in successful_vanilla) / len(successful_vanilla)

        if successful_rag:
            analysis["performance_comparison"]["rag_avg_time"] = sum(r["elapsed_time"] for r in successful_rag) / len(successful_rag)
            analysis["performance_comparison"]["avg_sources_retrieved"] = sum(r["num_sources"] for r in successful_rag) / len(successful_rag)

        # Summary
        analysis["summary"]["total_comparisons"] = len(results["comparisons"])
        analysis["summary"]["vanilla_responses"] = len(vanilla_responses)
        analysis["summary"]["rag_responses"] = len(rag_responses)

        return analysis

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
    parser = argparse.ArgumentParser(description="Compare vanilla LLM vs RAG responses")
    parser.add_argument("questions_file", help="Text file with questions (one per line)")
    parser.add_argument("--output", "-o", default="prompt_comparison.json", help="Output JSON file")
    parser.add_argument("--k-values", "-k", default="15", help="Comma-separated K values to test for RAG (e.g., 5,10,15)")
    parser.add_argument("--config", help="JSON file with config overrides")
    parser.add_argument("--analyze", action="store_true", help="Include analysis in output")

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
    print(f"Testing RAG with K values: {k_values}")
    print(f"Comparison: Vanilla LLM vs RAG")

    # Run comparison
    comparator = PromptComparator(config_overrides)
    results = comparator.compare_question_list(questions, k_values)

    # Add analysis if requested
    if args.analyze:
        results["analysis"] = comparator.analyze_results(results)

    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\\nResults saved to {args.output}")

    # Print summary
    total_tests = len(questions) * (1 + len(k_values))  # 1 vanilla + N rag tests per question
    successful_tests = sum(1 for comp in results["comparisons"] for resp in comp["responses"] if resp["success"])

    print(f"\\nSummary:")
    print(f"  Total tests: {total_tests}")
    print(f"  Successful: {successful_tests}")
    print(f"  Failed: {total_tests - successful_tests}")

    if args.analyze and "analysis" in results:
        analysis = results["analysis"]
        print(f"\\nPerformance Analysis:")
        print(f"  Vanilla success rate: {analysis['success_rates']['vanilla']:.1%}")
        print(f"  RAG success rate: {analysis['success_rates']['rag']:.1%}")

        if "vanilla_avg_time" in analysis["performance_comparison"]:
            print(f"  Vanilla avg time: {analysis['performance_comparison']['vanilla_avg_time']:.2f}s")
        if "rag_avg_time" in analysis["performance_comparison"]:
            print(f"  RAG avg time: {analysis['performance_comparison']['rag_avg_time']:.2f}s")

if __name__ == "__main__":
    main()