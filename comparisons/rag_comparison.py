#!/usr/bin/env python3
"""
RAG vs Raw Gemini Comparison Script

This script compares outputs from:
1. RAG pipeline (using your existing setup with Pinecone + Gemini)
2. Raw Gemini (without retrieval augmentation)
3. Bias-cleansed versions of both outputs for fair comparison

Usage:
    python rag_comparison.py "Your question here"
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Add the src/app directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from app.config import get_config, validate_config
from app.rag_service import RAGService
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from app.content_utils import extract_citation_titles_from_chunks, get_matched_references_for_text, format_reference_line
from app.prompts import MedicalPromptManager

# BiasCleanser class removed - now using template-based cleansing

def _add_citations_to_rag_response(self, rag_answer: str, source_docs: List) -> str:
    """
    Add citations to RAG response in clean Vancouver-style format
    
    Args:
        rag_answer: The original RAG response
        source_docs: List of source documents from retrieval
        
    Returns:
        RAG response with clean, numbered citations
    """
    import re
    
    try:
        # Extract citation titles from chunks
        citation_titles = extract_citation_titles_from_chunks(source_docs)
        
        enhanced_response = rag_answer
        
        if citation_titles:
            # Clean up the main response first
            # Remove phrases that indicate RAG retrieval
            enhanced_response = enhanced_response.replace("Based on the provided text, ", "")
            enhanced_response = enhanced_response.replace("The document states: ", "")
            enhanced_response = enhanced_response.replace("according to the documents", "")
            
            # Add references in clean format
            enhanced_response += "\n\n### References:\n\n"
            
            for i, citation in enumerate(citation_titles, 1):
                # Parse authors to get cleaner format
                authors = citation['authors']
                
                # Extract first author's last name
                if ',' in authors:
                    # Split by comma and get first author
                    author_parts = authors.split(',')[0].strip()
                    # Try to extract last name
                    if '.' in author_parts:
                        # Format like "Smith, J.A." -> extract "Smith"
                        first_author = author_parts.split('.')[0].strip()
                    else:
                        # Just use what we have
                        first_author = author_parts.split()[-1] if author_parts else "Unknown"
                else:
                    # No comma, try to get first word
                    first_author = authors.split()[0] if authors else "Unknown"
                
                # Clean up the title (remove any markdown formatting)
                title = citation['title'].replace('**', '').strip()
                
                # Format: Number. First Author et al. Title. Year.
                enhanced_response += f"{i}. {first_author} et al. {title}. {citation['year']}.\n"
        
        else:
            # Fallback to answer-based references if no citations found in chunks
            matched_refs = get_matched_references_for_text(rag_answer)
            if matched_refs:
                enhanced_response += "\n\n### References:\n\n"
                for i, m in enumerate(matched_refs, 1):
                    # Extract key info from reference
                    ref_text = format_reference_line(m['ref'])
                    # Simple format
                    enhanced_response += f"{i}. {ref_text}\n"
        
        return enhanced_response
        
    except Exception as e:
        print(f"Warning: Failed to add citations to RAG response: {e}")
        return rag_answer  # Return original if citation addition fails
    
class RAGComparison:
    """RAG vs Raw Gemini comparison with template-based prompts"""

    def __init__(self):
        # Load configuration
        self.config = get_config()
        validate_config(self.config)

        # Initialize prompt manager
        self.prompt_manager = MedicalPromptManager()

        # Initialize services
        self.rag_service = RAGService(self.config)

        self.raw_llm = ChatGoogleGenerativeAI(
            model=self.config['llm_model'],
            temperature=self.config['llm_temperature'],
            max_tokens=self.config['llm_max_tokens'],
            google_api_key=self.config['google_api_key']
        )

        print(f"Initialized comparison system with model: {self.config['llm_model']}")
        print(f"Retrieval k: {self.config['retrieval_k']}")
        print(f"Using template-based prompts")
    
    def _add_citations_to_rag_response(self, rag_answer: str, source_docs: List) -> str:
        """
        Add citations to RAG response, mimicking the main app's behavior
        
        Args:
            rag_answer: The original RAG response
            source_docs: List of source documents from retrieval
            
        Returns:
            RAG response with citations appended
        """
        try:
            # Extract citation titles from chunks (same as main app)
            citation_titles = extract_citation_titles_from_chunks(source_docs)
            
            enhanced_response = rag_answer
            
            if citation_titles:
                enhanced_response += "\n\n**Citations from Sources:**\n"
                for i, citation in enumerate(citation_titles, 1):
                    enhanced_response += f"{i}. **{citation['title']}**\n"
                    enhanced_response += f"   {citation['authors']} ({citation['year']})\n\n"
            else:
                # Fallback to answer-based references if no citations found in chunks
                matched_refs = get_matched_references_for_text(rag_answer)
                if matched_refs:
                    enhanced_response += "\n\n**References:**\n"
                    for m in matched_refs:
                        enhanced_response += f"Reference {m['number']}: {format_reference_line(m['ref'])}\n"
            
            return enhanced_response
            
        except Exception as e:
            print(f"Warning: Failed to add citations to RAG response: {e}")
            return rag_answer  # Return original if citation addition fails
    
    def get_rag_response(self, question: str) -> Tuple[str, str, List, float]:
        """Get response from RAG pipeline with and without citations"""
        start_time = time.time()
        
        try:
            result = self.rag_service.get_response(
                question,
                [],  # No chat history for one-shot
                retrieval_k=self.config['retrieval_k']
            )
            
            elapsed = time.time() - start_time
            answer = result["answer"]
            source_docs = result.get("source_documents", [])
            
            # Create enhanced response with citations (before bias cleansing)
            answer_with_citations = self._add_citations_to_rag_response(answer, source_docs)
            
            return answer, answer_with_citations, source_docs, elapsed
            
        except Exception as e:
            print(f"RAG error: {e}")
            error_msg = f"Error: {e}"
            return error_msg, error_msg, [], time.time() - start_time
    
    def get_raw_gemini_response(self, question: str) -> Tuple[str, float]:
        """Get response from raw Gemini without retrieval using template"""
        start_time = time.time()

        try:
            # Use vanilla template
            vanilla_prompt = self.prompt_manager.get_vanilla_prompt()
            formatted_prompt = vanilla_prompt.format(question=question)

            messages = [
                HumanMessage(content=formatted_prompt)
            ]

            response = self.raw_llm.invoke(messages)
            elapsed = time.time() - start_time

            return response.content.strip(), elapsed

        except Exception as e:
            print(f"Raw Gemini error: {e}")
            return f"Error: {e}", time.time() - start_time
    
    def cleanse_responses(self, rag_text: str, raw_text: str) -> Tuple[str, str]:
        """Apply bias cleansing to both responses using template"""
        print("Applying bias cleansing...")

        try:
            # Use cleansing template
            cleansing_prompt = self.prompt_manager.get_cleansing_prompt()

            # Clean RAG response
            rag_formatted = cleansing_prompt.format(context_type="medical/therapy", text=rag_text)
            rag_response = self.raw_llm.invoke([HumanMessage(content=rag_formatted)])
            rag_cleaned = rag_response.content.strip()

            # Clean vanilla response
            raw_formatted = cleansing_prompt.format(context_type="medical/therapy", text=raw_text)
            raw_response = self.raw_llm.invoke([HumanMessage(content=raw_formatted)])
            raw_cleaned = raw_response.content.strip()

            return rag_cleaned, raw_cleaned

        except Exception as e:
            print(f"Warning: Cleansing failed: {e}")
            return rag_text, raw_text  # Return originals if cleansing fails
    
    def compare_responses(self, question: str, save_results: bool = True) -> Dict:
        """Run the full comparison pipeline"""
        print(f"\nQuestion: {question}")
        print("=" * 80)

        # Get RAG response
        print("\nGetting RAG response...")
        rag_answer, rag_answer_with_citations, source_docs, rag_time = self.get_rag_response(question)
        
        # Get raw Gemini response
        print("Getting raw Gemini response...")
        raw_answer, raw_time = self.get_raw_gemini_response(question)
        
        # Cleanse responses (using the version with citations for RAG)
        rag_cleaned, raw_cleaned = self.cleanse_responses(rag_answer_with_citations, raw_answer)
        
        # Compile results with enhanced logging structure
        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "comparison_type": "RAG_vs_Raw_Gemini",
                "bias_cleansing_applied": True,
                "citations_added_to_rag": True
            },
            "responses": {
                "rag_original": {
                    "method": "RAG (Retrieval Augmented Generation) - Original",
                    "description": "Response generated using RAG pipeline with Pinecone document retrieval (no citations added)",
                    "content": rag_answer,
                    "response_time": rag_time,
                    "sources_count": len(source_docs),
                    "sources": [doc.metadata.get('source', 'Unknown') for doc in source_docs[:5]],  # Top 5 sources
                    "has_retrieval": True,
                    "bias_cleansed": False,
                    "has_citations": False
                },
                "rag_with_citations": {
                    "method": "RAG (Retrieval Augmented Generation) - With Citations",
                    "description": "RAG response with citations from references.json appended (before bias cleansing)",
                    "content": rag_answer_with_citations,
                    "response_time": rag_time,
                    "sources_count": len(source_docs),
                    "sources": [doc.metadata.get('source', 'Unknown') for doc in source_docs[:5]],
                    "has_retrieval": True,
                    "bias_cleansed": False,
                    "has_citations": True
                },
                "rag_cleansed": {
                    "method": "RAG (Retrieval Augmented Generation) - With Citations, Bias Cleansed",
                    "description": "RAG response with citations, then source indicators removed for unbiased comparison",
                    "content": rag_cleaned,
                    "response_time": rag_time,  # Same as original since cleansing is fast
                    "sources_count": len(source_docs),
                    "sources": [doc.metadata.get('source', 'Unknown') for doc in source_docs[:5]],
                    "has_retrieval": True,
                    "bias_cleansed": True,
                    "has_citations": True
                },
                "raw_gemini_original": {
                    "method": "Raw Gemini (No Retrieval)",
                    "description": "Response generated using direct Gemini API call without document retrieval",
                    "content": raw_answer,
                    "response_time": raw_time,
                    "sources_count": 0,
                    "sources": [],
                    "has_retrieval": False,
                    "bias_cleansed": False,
                    "has_citations": False
                },
                "raw_gemini_cleansed": {
                    "method": "Raw Gemini (No Retrieval) - Bias Cleansed",
                    "description": "Raw Gemini response with source indicators removed for unbiased comparison",
                    "content": raw_cleaned,
                    "response_time": raw_time,  # Same as original since cleansing is fast
                    "sources_count": 0,
                    "sources": [],
                    "has_retrieval": False,
                    "bias_cleansed": True,
                    "has_citations": False
                }
            },
            "comparison_metrics": {
                "rag_vs_raw_time_diff": rag_time - raw_time,
                "rag_has_sources": len(source_docs) > 0,
                "total_sources_used": len(source_docs)
            },
            "config": {
                "model": self.config['llm_model'],
                "temperature": self.config['llm_temperature'],
                "max_tokens": self.config['llm_max_tokens'],
                "retrieval_k": self.config['retrieval_k'],
                "bias_cleansing_model": self.config['llm_model']
            }
        }
        
        # Display results
        self._display_results(results)
        
        # Save results if requested
        if save_results:
            self._save_results(results)
        
        return results
    
    def _display_results(self, results: Dict):
        """Display comparison results in a formatted way with clear labeling"""
        print("\n" + "="*80)
        print("📊 COMPARISON RESULTS")
        print("="*80)
        
        # Extract data from new structure
        rag_orig = results['responses']['rag_original']
        rag_with_citations = results['responses']['rag_with_citations']
        rag_clean = results['responses']['rag_cleansed']
        raw_orig = results['responses']['raw_gemini_original']
        raw_clean = results['responses']['raw_gemini_cleansed']
        
        print(f"\n⏱️  Response Times:")
        print(f"   📚 RAG (with retrieval): {rag_orig['response_time']:.2f}s")
        print(f"   🤖 Raw Gemini (no retrieval): {raw_orig['response_time']:.2f}s")
        
        if rag_orig['sources_count'] > 0:
            print(f"\n📚 RAG Sources ({rag_orig['sources_count']} total):")
            for i, source in enumerate(rag_orig['sources'][:3], 1):
                print(f"   {i}. {source}")
            if len(rag_orig['sources']) > 3:
                print(f"   ... and {len(rag_orig['sources']) - 3} more")
        
        print(f"\n" + "="*80)
        print("📝 RAG RESPONSE PROGRESSION")
        print("="*80)
        
        print(f"\n📚 RAG ORIGINAL (NO CITATIONS):")
        print("🔸 Source: RAG pipeline with Pinecone document retrieval")
        print("-" * 60)
        print(rag_orig['content'])
        
        print(f"\n📚 RAG WITH CITATIONS (BEFORE BIAS CLEANSING):")
        print("🔸 Source: RAG pipeline + citations from references.json")
        print("-" * 60)
        print(rag_with_citations['content'])
        
        print(f"\n" + "="*80)
        print("📝 RAW GEMINI RESPONSE")
        print("="*80)
        
        print(f"\n🤖 RAW GEMINI RESPONSE (ORIGINAL - NO RETRIEVAL):")
        print("🔸 Source: Direct Gemini API call without document retrieval")
        print("-" * 60)
        print(raw_orig['content'])
        
        print(f"\n" + "="*80)
        print("🧹 BIAS-CLEANSED RESPONSES (SOURCE INDICATORS REMOVED)")
        print("="*80)
        print("💡 These responses have been processed to remove indicators of their source method")
        print("   for unbiased content comparison.")
        
        print(f"\nResponse A (CLEANSED - ORIGINALLY RAW GEMINI):")
        print("🔸 Source indicators removed from raw Gemini response")
        print("-" * 60)
        print(raw_clean['content'])
        
        print(f"\nResponse B (CLEANSED - ORIGINALLY RAG WITH CITATIONS):")
        print("🔸 Source indicators removed from RAG response (includes citations)")
        print("-" * 60)
        print(rag_clean['content'])
        
        print(f"\n" + "="*80)
        print("📋 SUMMARY")
        print("="*80)
        print(f"📝 Question: {results['metadata']['question']}")
        print(f"⏱️  Raw Gemini time: {raw_orig['response_time']:.2f}s")
        print(f"⏱️  RAG time: {rag_orig['response_time']:.2f}s")
        print(f"📚 RAG sources used: {rag_orig['sources_count']}")
        print(f"🤖 Model used: {results['config']['model']}")
        print(f"🧹 Bias cleansing applied: {results['metadata']['bias_cleansing_applied']}")
        print(f"📖 Citations added to RAG: {results['metadata']['citations_added_to_rag']}")
        
        print(f"\n💡 Which cleansed response do you prefer? (A = originally Raw Gemini, B = originally RAG with citations)")
    
    def _save_results(self, results: Dict):
        """Save results to a JSON file with detailed logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rag_comparison_{timestamp}.json"
        
        # Add a detailed summary for easier analysis
        results['analysis_summary'] = self._create_analysis_summary(results)
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {filename}")
            
            # Also create a human-readable summary
            summary_filename = f"rag_comparison_summary_{timestamp}.txt"
            self._create_human_readable_summary(results, summary_filename)
            
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
    
    def _create_analysis_summary(self, results: Dict) -> Dict:
        """Create a structured summary for analysis"""
        rag_orig = results['responses']['rag_original']
        rag_with_citations = results['responses']['rag_with_citations']
        rag_clean = results['responses']['rag_cleansed']
        raw_orig = results['responses']['raw_gemini_original']
        raw_clean = results['responses']['raw_gemini_cleansed']
        
        return {
            "question": results['metadata']['question'],
            "timestamp": results['metadata']['timestamp'],
            "response_types": {
                "rag_original": {
                    "method": rag_orig['method'],
                    "has_retrieval": rag_orig['has_retrieval'],
                    "bias_cleansed": rag_orig['bias_cleansed'],
                    "has_citations": rag_orig['has_citations'],
                    "sources_count": rag_orig['sources_count'],
                    "response_time": rag_orig['response_time']
                },
                "rag_with_citations": {
                    "method": rag_with_citations['method'],
                    "has_retrieval": rag_with_citations['has_retrieval'],
                    "bias_cleansed": rag_with_citations['bias_cleansed'],
                    "has_citations": rag_with_citations['has_citations'],
                    "sources_count": rag_with_citations['sources_count'],
                    "response_time": rag_with_citations['response_time']
                },
                "rag_cleansed": {
                    "method": rag_clean['method'],
                    "has_retrieval": rag_clean['has_retrieval'],
                    "bias_cleansed": rag_clean['bias_cleansed'],
                    "has_citations": rag_clean['has_citations'],
                    "sources_count": rag_clean['sources_count'],
                    "response_time": rag_clean['response_time']
                },
                "raw_gemini_original": {
                    "method": raw_orig['method'],
                    "has_retrieval": raw_orig['has_retrieval'],
                    "bias_cleansed": raw_orig['bias_cleansed'],
                    "has_citations": raw_orig['has_citations'],
                    "response_time": raw_orig['response_time']
                },
                "raw_gemini_cleansed": {
                    "method": raw_clean['method'],
                    "has_retrieval": raw_clean['has_retrieval'],
                    "bias_cleansed": raw_clean['bias_cleansed'],
                    "has_citations": raw_clean['has_citations'],
                    "response_time": raw_clean['response_time']
                }
            },
            "performance_metrics": {
                "rag_time": rag_orig['response_time'],
                "raw_gemini_time": raw_orig['response_time'],
                "time_difference": results['comparison_metrics']['rag_vs_raw_time_diff'],
                "rag_used_sources": results['comparison_metrics']['rag_has_sources'],
                "total_sources": results['comparison_metrics']['total_sources_used']
            }
        }
    
    def _create_human_readable_summary(self, results: Dict, filename: str):
        """Create a human-readable summary file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("RAG vs Raw Gemini Comparison Summary\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Question: {results['metadata']['question']}\n")
                f.write(f"Timestamp: {results['metadata']['timestamp']}\n")
                f.write(f"Model: {results['config']['model']}\n\n")
                
                f.write("RESPONSE TYPES:\n")
                f.write("-" * 20 + "\n")
                f.write("1. RAG Original - Generated with document retrieval (no citations)\n")
                f.write("2. RAG With Citations - RAG response + citations from references.json\n")
                f.write("3. RAG Cleansed - RAG with citations, bias indicators removed\n")
                f.write("4. Raw Gemini Original - Generated without document retrieval\n")
                f.write("5. Raw Gemini Cleansed - Raw response with bias indicators removed\n\n")
                
                f.write("PERFORMANCE METRICS:\n")
                f.write("-" * 20 + "\n")
                rag_time = results['responses']['rag_original']['response_time']
                raw_time = results['responses']['raw_gemini_original']['response_time']
                f.write(f"RAG Response Time: {rag_time:.2f}s\n")
                f.write(f"Raw Gemini Response Time: {raw_time:.2f}s\n")
                f.write(f"Time Difference: {rag_time - raw_time:.2f}s\n")
                f.write(f"RAG Sources Used: {results['responses']['rag_original']['sources_count']}\n\n")
                
                f.write("RESPONSE CONTENTS:\n")
                f.write("-" * 20 + "\n")
                
                for key, response in results['responses'].items():
                    f.write(f"\n{response['method'].upper()}:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Description: {response['description']}\n")
                    f.write(f"Has Retrieval: {response['has_retrieval']}\n")
                    f.write(f"Bias Cleansed: {response['bias_cleansed']}\n")
                    f.write(f"Response Time: {response['response_time']:.2f}s\n")
                    if response['sources_count'] > 0:
                        f.write(f"Sources: {', '.join(response['sources'][:3])}\n")
                    f.write(f"Content:\n{response['content']}\n")
                    f.write("\n" + "="*60 + "\n")
            
            print(f"📄 Human-readable summary saved to: {filename}")
            
        except Exception as e:
            print(f"❌ Failed to create summary: {e}")
    
    def batch_compare(self, questions: List[str], save_individual: bool = True):
        """Compare multiple questions in batch"""
        print(f"\n🔄 Running batch comparison for {len(questions)} questions...")
        
        all_results = []
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] Processing question...")
            result = self.compare_responses(question, save_individual)
            all_results.append(result)
        
        # Save batch summary
        batch_summary = {
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(questions),
            "results": all_results
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_filename = f"rag_batch_comparison_{timestamp}.json"
        
        try:
            with open(batch_filename, 'w', encoding='utf-8') as f:
                json.dump(batch_summary, f, indent=2, ensure_ascii=False)
            print(f"\n📊 Batch results saved to: {batch_filename}")
        except Exception as e:
            print(f"❌ Failed to save batch results: {e}")


def main():
    """Simple main function for RAG vs Vanilla comparison"""
    if len(sys.argv) < 2:
        print("Usage: python rag_comparison.py \"Your question here\"")
        print("\nExample:")
        print("  python rag_comparison.py \"How should I interpret R-time prolongation on TEG?\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])

    try:
        # Initialize comparison system
        comparison = RAGComparison()

        print(f"\n📋 Question: {question}")
        print("=" * 80)

        # Run comparison
        results = comparison.compare_responses(question)

        print(f"\n✅ Comparison completed successfully!")
        print(f"📊 Results saved to: {results.get('output_file', 'comparison output file')}")

    except Exception as e:
        print(f"❌ Error running comparison: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
