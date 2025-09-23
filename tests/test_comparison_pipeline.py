"""
Comparison pipeline for testing all questions from CSV through both vanilla LLM and RAG system
Logs both responses after running through the cleansing script
"""

import pandas as pd
import csv
import uuid
from datetime import datetime
from typing import List, Dict, Optional
import os
import sys
import random

# Add src to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'comparisons'))

from app.config import get_config, validate_config
from rag_comparison import RAGComparison
from dotenv import load_dotenv

load_dotenv()

class ComparisonTestPipeline:
    """Pipeline for testing vanilla vs RAG with cleansing on clinical questions"""

    def __init__(self, config_override: Optional[Dict] = None):
        """Initialize the comparison testing pipeline"""
        self.config = get_config()
        if config_override:
            self.config.update(config_override)
        validate_config(self.config)

        self.comparison_service = RAGComparison()
        self.session_id = str(uuid.uuid4())[:8]

    def load_questions_from_csv(self, csv_path: str) -> List[Dict]:
        """Load and parse questions from the CSV file"""
        questions = []

        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)

            for row_idx, row in enumerate(reader):
                # Skip header row and extract questions from columns
                person_name = row.get('Name', f'Person_{row_idx}')
                experience = row.get('How many years of clinical experience do you have? \n(years since medical school graduation)', 'Unknown')

                # Extract all question columns (Question 1 through Question 6)
                for i in range(1, 7):
                    question_col = f'Question {i}'
                    if question_col in row and row[question_col].strip():
                        questions.append({
                            'person_name': person_name,
                            'experience_years': experience,
                            'question_number': i,
                            'question_text': row[question_col].strip(),
                            'row_index': row_idx
                        })

        return questions

    def sample_questions(self, questions: List[Dict], sample_size: Optional[int] = None) -> List[Dict]:
        """Sample questions for testing"""
        if sample_size is None or sample_size >= len(questions):
            return questions

        return random.sample(questions, sample_size)

    def run_single_question_comparison(self, question_data: Dict) -> Dict:
        """Run a single question through both vanilla and RAG systems with cleansing"""
        start_time = datetime.now()

        try:
            # Get RAG response
            rag_answer, rag_answer_with_citations, source_docs, rag_time = self.comparison_service.get_rag_response(question_data['question_text'])

            # Get raw Gemini response
            raw_answer, raw_time = self.comparison_service.get_raw_gemini_response(question_data['question_text'])

            # Cleanse responses
            rag_cleaned, raw_cleaned = self.comparison_service.cleanse_responses(rag_answer_with_citations, raw_answer)

            end_time = datetime.now()
            elapsed_time = (end_time - start_time).total_seconds()

            return {
                'success': True,
                'vanilla_original': raw_answer,
                'vanilla_cleansed': raw_cleaned,
                'rag_original': rag_answer,
                'rag_with_citations': rag_answer_with_citations,
                'rag_cleansed': rag_cleaned,
                'vanilla_response_time': raw_time,
                'rag_response_time': rag_time,
                'total_elapsed_time': elapsed_time,
                'num_source_docs': len(source_docs),
                'sources': [doc.metadata.get('source', 'Unknown') for doc in source_docs[:5]],
                'timestamp': start_time.isoformat(),
                'error': None
            }

        except Exception as e:
            end_time = datetime.now()
            elapsed_time = (end_time - start_time).total_seconds()

            return {
                'success': False,
                'vanilla_original': "",
                'vanilla_cleansed': "",
                'rag_original': "",
                'rag_with_citations': "",
                'rag_cleansed': "",
                'vanilla_response_time': 0,
                'rag_response_time': 0,
                'total_elapsed_time': elapsed_time,
                'num_source_docs': 0,
                'sources': [],
                'timestamp': start_time.isoformat(),
                'error': str(e)
            }

    def run_pipeline(self, csv_path: str, output_path: str, sample_size: Optional[int] = None) -> str:
        """
        Run the complete comparison testing pipeline

        Args:
            csv_path: Path to input CSV with questions
            output_path: Path for output CSV with results
            sample_size: Number of questions to sample (None for all)

        Returns:
            Path to the output CSV file
        """
        print(f"Loading questions from {csv_path}...")
        questions = self.load_questions_from_csv(csv_path)
        print(f"Found {len(questions)} total questions")

        if sample_size:
            questions = self.sample_questions(questions, sample_size)
            print(f"Sampled {len(questions)} questions for testing")

        results = []

        print(f"Running vanilla vs RAG comparison pipeline")
        print(f"Session ID: {self.session_id}")
        print("Note: Both responses will be cleansed to remove bias indicators")

        for i, question_data in enumerate(questions, 1):
            print(f"Processing question {i}/{len(questions)}: {question_data['person_name']} Q{question_data['question_number']}")

            # Run the question through both systems
            comparison_result = self.run_single_question_comparison(question_data)

            # Combine question data with results
            result_row = {
                'session_id': self.session_id,
                'person_name': question_data['person_name'],
                'experience_years': question_data['experience_years'],
                'question_number': question_data['question_number'],
                'question_text': question_data['question_text'],

                # Vanilla LLM responses
                'vanilla_original': comparison_result['vanilla_original'],
                'vanilla_cleansed': comparison_result['vanilla_cleansed'],
                'vanilla_response_time': comparison_result['vanilla_response_time'],

                # RAG responses
                'rag_original': comparison_result['rag_original'],
                'rag_with_citations': comparison_result['rag_with_citations'],
                'rag_cleansed': comparison_result['rag_cleansed'],
                'rag_response_time': comparison_result['rag_response_time'],
                'num_source_docs': comparison_result['num_source_docs'],
                'sources_used': '; '.join(comparison_result['sources'][:5]),  # Top 5 sources

                # Meta information
                'success': comparison_result['success'],
                'total_elapsed_time': comparison_result['total_elapsed_time'],
                'timestamp': comparison_result['timestamp'],
                'error': comparison_result['error']
            }

            results.append(result_row)

            # Print status
            if comparison_result['success']:
                print(f"  Success - Vanilla: {comparison_result['vanilla_response_time']:.2f}s, RAG: {comparison_result['rag_response_time']:.2f}s")
            else:
                print(f"  Error: {comparison_result['error']}")

        # Save results to CSV
        print(f"Saving results to {output_path}...")
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False, encoding='utf-8')

        # Print summary
        successful = sum(1 for r in results if r['success'])
        avg_vanilla_time = sum(r['vanilla_response_time'] for r in results if r['success']) / max(successful, 1)
        avg_rag_time = sum(r['rag_response_time'] for r in results if r['success']) / max(successful, 1)
        avg_total_time = sum(r['total_elapsed_time'] for r in results if r['success']) / max(successful, 1)

        print(f"\nComparison pipeline complete!")
        print(f"Total questions: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")
        print(f"Average vanilla response time: {avg_vanilla_time:.2f}s")
        print(f"Average RAG response time: {avg_rag_time:.2f}s")
        print(f"Average total time per question: {avg_total_time:.2f}s")
        print(f"Results saved to: {output_path}")

        return output_path


def main():
    """Main function for running the comparison pipeline"""
    # Configuration
    INPUT_CSV = r"C:\Users\agish\Desktop\RAG-Therapy\OneDrive_1_5-18-2025\Thera-RAG\comparisons\VHA Guideline LLM Survey (Responses).csv"
    OUTPUT_DIR = r"C:\Users\agish\Desktop\RAG-Therapy\OneDrive_1_5-18-2025\Thera-RAG\tests\results"

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"vanilla_vs_rag_comparison_{timestamp}.csv")

    # Initialize pipeline
    pipeline = ComparisonTestPipeline()

    # Run pipeline - modify these parameters as needed
    pipeline.run_pipeline(
        csv_path=INPUT_CSV,
        output_path=output_file,
        sample_size=None  # Set to a number for sampling, None for all questions
    )


if __name__ == "__main__":
    main()