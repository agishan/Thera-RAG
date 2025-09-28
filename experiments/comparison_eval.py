"""
RAG vs Vanilla Response Comparison Script

This script compares RAG responses (with retrieved chunks and citations)
against vanilla LLM responses (no context) for clinical questions.

Input: CSV with questions
Output: CSV with RAG responses, vanilla responses, and cleansed versions
"""

import pandas as pd
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging

# Add the inference app to the path to import components
inference_path = Path(__file__).parent.parent / "apps" / "inference" / "src"
sys.path.insert(0, str(inference_path))

# Import modules individually to handle relative import issues
import importlib.util

def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load modules
config_module = load_module_from_path("config", inference_path / "config.py")
get_config = config_module.get_config
validate_config = config_module.validate_config

# For content_utils
content_utils = load_module_from_path("content_utils", inference_path / "content_utils.py")
extract_citation_titles_from_chunks = content_utils.extract_citation_titles_from_chunks

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Simplified RAG service for comparison script
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

class SimplifiedRAGService:
    """Simplified RAG service for comparison evaluation"""

    def __init__(self, config):
        self.config = config

        # Initialize Pinecone
        self.pc = Pinecone(api_key=config['pinecone_api_key'])
        self.index = self.pc.Index(config['pinecone_index_name'])
        self.embedder = SentenceTransformer(config['embedding_model'])

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=config['llm_model'],
            temperature=config['llm_temperature'],
            max_tokens=config['llm_max_tokens'],
            google_api_key=config['google_api_key']
        )

        # RAG prompt
        self.rag_prompt = """You are a knowledgeable medical assistant with expertise in clinical guidelines and viscoelastic testing.

Use the following pieces of context to answer the question at the end. Provide concise, evidence-based answers to questions about therapy approaches and clinical practice. Be accurate and precise. If you don't know the answer based on the provided context, just say that you don't know.

{context}

Question: {question}
Answer:"""

    def get_response(self, question: str, chat_history: List = None, retrieval_k: int = 15):
        """Get RAG response with retrieved chunks"""

        # Generate embedding for the question
        query_embedding = self.embedder.encode(question).tolist()

        # Query Pinecone
        search_results = self.index.query(
            vector=query_embedding,
            top_k=retrieval_k,
            namespace=self.config['pinecone_namespace'],
            include_metadata=True
        )

        # Convert to Document objects
        source_docs = []
        for match in search_results['matches']:
            # Add score to metadata
            metadata = match['metadata'].copy()
            metadata['score'] = match['score']

            doc = Document(
                page_content=match['metadata'].get('text', ''),
                metadata=metadata
            )
            source_docs.append(doc)

        # Format context
        context = "\n\n".join([doc.page_content for doc in source_docs])

        # Generate response
        formatted_prompt = self.rag_prompt.format(context=context, question=question)
        response = self.llm.invoke(formatted_prompt)

        return {
            "answer": response.content,
            "source_documents": source_docs
        }


class ComparisonEvaluator:
    """Evaluates RAG vs Vanilla responses with cleansing"""

    def __init__(self, config: dict, delay_between_calls: float = 30.0):
        """
        Initialize the comparison evaluator

        Args:
            config: Configuration dictionary from get_config()
            delay_between_calls: Seconds to wait between API calls for rate limiting
        """
        self.config = config
        self.delay_between_calls = delay_between_calls
        self.last_api_call = None

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize simplified RAG service
        self.rag_service = SimplifiedRAGService(config)

        # Initialize vanilla LLM (same model, no retrieval)
        self.vanilla_llm = ChatGoogleGenerativeAI(
            model=config['llm_model'],
            temperature=config['llm_temperature'],
            max_tokens=config['llm_max_tokens'],
            google_api_key=config['google_api_key']
        )

        # Setup prompts
        self._setup_prompts()

    def _setup_prompts(self):
        """Setup vanilla and cleansing prompts"""

        # Vanilla prompt (no context)
        self.vanilla_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are a knowledgeable medical assistant with expertise in clinical guidelines and viscoelastic testing.

Provide a concise, evidence-based answer to the following medical question. Base your response on established clinical knowledge and best practices.

Question: {question}
Answer:"""
        )

        # Cleansing prompt to remove bias and overly confident language
        self.cleansing_prompt = PromptTemplate(
            input_variables=["original_response"],
            template="""Please review and improve the following medical response by:
1. Removing overly confident or absolute statements
2. Adding appropriate medical disclaimers where needed
3. Ensuring balanced, evidence-based language
4. Maintaining clinical accuracy while being appropriately cautious

Original Response:
{original_response}

Improved Response:"""
        )

    def _wait_for_rate_limit(self):
        """Wait appropriate time between API calls"""
        if self.last_api_call:
            elapsed = (datetime.now() - self.last_api_call).total_seconds()
            if elapsed < self.delay_between_calls:
                wait_time = self.delay_between_calls - elapsed
                self.logger.info(f"Rate limiting: waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)

    def get_rag_response(self, question: str, k_chunks: int = 15, k_citation_chunks: int = 10) -> Dict:
        """
        Get RAG response with retrieved chunks and citations

        Args:
            question: The clinical question
            k_chunks: Number of chunks to retrieve
            k_citation_chunks: Number of chunks to extract citations from

        Returns:
            Dictionary with response data and metadata
        """
        try:
            self._wait_for_rate_limit()
            start_time = time.time()

            # Get RAG response
            result = self.rag_service.get_response(
                question=question,
                chat_history=[],
                retrieval_k=k_chunks
            )

            # Extract citations from top citation chunks
            source_docs = result.get("source_documents", [])
            citation_titles = extract_citation_titles_from_chunks(source_docs[:k_citation_chunks])

            # Format citations as additional context (for reference)
            citations_text = ""
            if citation_titles:
                citations_text = "\n\nRelevant Citations:\n"
                for i, citation in enumerate(citation_titles, 1):
                    citations_text += f"{i}. {citation['title']} - {citation['authors']} ({citation['year']})\n"

            elapsed_time = time.time() - start_time
            self.last_api_call = datetime.now()

            return {
                "response": result["answer"],
                "retrieved_chunks": len(source_docs),
                "citation_chunks": len(citation_titles),
                "citations_found": citations_text.strip(),
                "response_time": elapsed_time,
                "error": None
            }

        except Exception as e:
            self.logger.error(f"RAG response error: {str(e)}")
            return {
                "response": "",
                "retrieved_chunks": 0,
                "citation_chunks": 0,
                "citations_found": "",
                "response_time": 0,
                "error": str(e)
            }

    def get_vanilla_response(self, question: str) -> Dict:
        """
        Get vanilla response (no context)

        Args:
            question: The clinical question

        Returns:
            Dictionary with response data and metadata
        """
        try:
            self._wait_for_rate_limit()
            start_time = time.time()

            # Format prompt and get response
            formatted_prompt = self.vanilla_prompt.format(question=question)
            response = self.vanilla_llm.invoke(formatted_prompt)

            elapsed_time = time.time() - start_time
            self.last_api_call = datetime.now()

            return {
                "response": response.content,
                "response_time": elapsed_time,
                "error": None
            }

        except Exception as e:
            self.logger.error(f"Vanilla response error: {str(e)}")
            return {
                "response": "",
                "response_time": 0,
                "error": str(e)
            }

    def cleanse_response(self, response: str) -> Dict:
        """
        Cleanse response to remove bias and overconfident language

        Args:
            response: Original response text

        Returns:
            Dictionary with cleansed response and metadata
        """
        try:
            self._wait_for_rate_limit()
            start_time = time.time()

            # Format prompt and get cleansed response
            formatted_prompt = self.cleansing_prompt.format(original_response=response)
            cleansed = self.vanilla_llm.invoke(formatted_prompt)

            elapsed_time = time.time() - start_time
            self.last_api_call = datetime.now()

            return {
                "cleansed_response": cleansed.content,
                "cleansing_time": elapsed_time,
                "error": None
            }

        except Exception as e:
            self.logger.error(f"Cleansing error: {str(e)}")
            return {
                "cleansed_response": response,  # Return original if cleansing fails
                "cleansing_time": 0,
                "error": str(e)
            }

    def process_questions(self, input_df: pd.DataFrame,
                         default_k_chunks: int = 15,
                         default_k_citation_chunks: int = 10) -> pd.DataFrame:
        """
        Process all questions in the dataframe

        Args:
            input_df: DataFrame with 'question' column and optional k_chunks, k_citation_chunks
            default_k_chunks: Default number of chunks to retrieve
            default_k_citation_chunks: Default number of citation chunks

        Returns:
            DataFrame with additional response columns
        """
        results = []
        total_questions = len(input_df)

        for idx, row in input_df.iterrows():
            question = row['question']
            k_chunks = row.get('k_chunks', default_k_chunks)
            k_citation_chunks = row.get('k_citation_chunks', default_k_citation_chunks)

            self.logger.info(f"Processing question {idx + 1}/{total_questions}: {question[:50]}...")

            # Get RAG response
            rag_result = self.get_rag_response(question, k_chunks, k_citation_chunks)

            # Get vanilla response
            vanilla_result = self.get_vanilla_response(question)

            # Cleanse both responses
            rag_cleansed = self.cleanse_response(rag_result["response"]) if rag_result["response"] else {"cleansed_response": "", "cleansing_time": 0, "error": "No RAG response to cleanse"}
            vanilla_cleansed = self.cleanse_response(vanilla_result["response"]) if vanilla_result["response"] else {"cleansed_response": "", "cleansing_time": 0, "error": "No vanilla response to cleanse"}

            # Compile results
            result_row = {
                **row.to_dict(),  # Original columns
                "rag_response": rag_result["response"],
                "vanilla_response": vanilla_result["response"],
                "rag_response_cleaned": rag_cleansed["cleansed_response"],
                "vanilla_response_cleaned": vanilla_cleansed["cleansed_response"],
                "retrieved_chunks": rag_result["retrieved_chunks"],
                "citation_chunks": rag_result["citation_chunks"],
                "citations_found": rag_result["citations_found"],
                "rag_response_time": rag_result["response_time"],
                "vanilla_response_time": vanilla_result["response_time"],
                "rag_cleansing_time": rag_cleansed["cleansing_time"],
                "vanilla_cleansing_time": vanilla_cleansed["cleansing_time"],
                "error_rag": rag_result["error"],
                "error_vanilla": vanilla_result["error"],
                "error_rag_cleansing": rag_cleansed["error"],
                "error_vanilla_cleansing": vanilla_cleansed["error"],
                "processed_at": datetime.now().isoformat()
            }

            results.append(result_row)

            # Progress update
            if (idx + 1) % 5 == 0:
                self.logger.info(f"Completed {idx + 1}/{total_questions} questions")

        return pd.DataFrame(results)


def run_comparison(input_csv: str,
                  output_csv: str,
                  k_chunks: int = 15,
                  k_citation_chunks: int = 10,
                  delay_between_calls: float = 30.0):
    """
    Main function to run the comparison evaluation

    Args:
        input_csv: Path to input CSV file with questions
        output_csv: Path to output CSV file for results
        k_chunks: Default number of chunks to retrieve
        k_citation_chunks: Default number of citation chunks
        delay_between_calls: Seconds between API calls for rate limiting
    """
    # Load configuration
    config = get_config()
    validate_config(config)

    # Load input data
    try:
        df = pd.read_csv(input_csv)

        # Handle different CSV formats
        if 'question' in df.columns:
            # Standard format
            pass
        elif 'Question_Text' in df.columns:
            # VHA format - rename column and filter out empty rows
            df = df[df['Question_Text'].notna() & (df['Question_Text'] != '')]
            df = df.rename(columns={'Question_Text': 'question'})
            print(f"Detected VHA questions format, using Question_Text column")
        else:
            raise ValueError("Input CSV must contain either 'question' or 'Question_Text' column")

        print(f"Loaded {len(df)} questions from {input_csv}")

    except Exception as e:
        print(f"Error loading input CSV: {e}")
        return

    # Initialize evaluator
    evaluator = ComparisonEvaluator(config, delay_between_calls)

    # Process questions
    print(f"Starting comparison evaluation...")
    print(f"Rate limiting: {delay_between_calls} seconds between API calls")
    print(f"Estimated total time: {len(df) * delay_between_calls * 4 / 60:.1f} minutes")  # 4 API calls per question

    try:
        results_df = evaluator.process_questions(df, k_chunks, k_citation_chunks)

        # Save results
        results_df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

        # Print summary
        success_rag = sum(1 for err in results_df['error_rag'] if err is None)
        success_vanilla = sum(1 for err in results_df['error_vanilla'] if err is None)

        print(f"\nSummary:")
        print(f"Total questions processed: {len(results_df)}")
        print(f"Successful RAG responses: {success_rag}/{len(results_df)}")
        print(f"Successful Vanilla responses: {success_vanilla}/{len(results_df)}")
        print(f"Average RAG response time: {results_df['rag_response_time'].mean():.2f}s")
        print(f"Average Vanilla response time: {results_df['vanilla_response_time'].mean():.2f}s")

    except Exception as e:
        print(f"Error during processing: {e}")
        return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare RAG vs Vanilla responses")
    parser.add_argument("input_csv", help="Path to input CSV file with questions")
    parser.add_argument("output_csv", help="Path to output CSV file for results")
    parser.add_argument("--k-chunks", type=int, default=15, help="Number of chunks to retrieve (default: 15)")
    parser.add_argument("--k-citation-chunks", type=int, default=10, help="Number of citation chunks (default: 10)")
    parser.add_argument("--delay", type=float, default=30.0, help="Delay between API calls in seconds (default: 30.0)")

    args = parser.parse_args()

    run_comparison(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        k_chunks=args.k_chunks,
        k_citation_chunks=args.k_citation_chunks,
        delay_between_calls=args.delay
    )