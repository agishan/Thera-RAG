"""
Testing pipeline for RAG system with clinical questions
Loads questions from CSV, runs through RAG system, saves results
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

from app.config import get_config, validate_config
from app.rag_service import RAGService
from dotenv import load_dotenv

load_dotenv()

class RAGTestPipeline:
    """Pipeline for testing RAG system with clinical questions"""

    def __init__(self, config_override: Optional[Dict] = None):
        """Initialize the testing pipeline"""
        self.config = get_config()
        if config_override:
            self.config.update(config_override)
        validate_config(self.config)

        self.rag_service = RAGService(self.config)
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

    def run_single_question(self, question_data: Dict, retrieval_k: int = 15,
                          prompt_type: str = "medical_rag") -> Dict:
        """Run a single question through the RAG system"""
        start_time = datetime.now()

        try:
            # Get response from RAG service
            result = self.rag_service.get_response(
                question=question_data['question_text'],
                chat_history=[],  # No conversation history for testing
                retrieval_k=retrieval_k,
                prompt_type=prompt_type
            )

            end_time = datetime.now()
            elapsed_time = (end_time - start_time).total_seconds()

            # Extract relevant information
            answer = result.get("answer", "")
            source_docs = result.get("source_documents", [])

            return {
                'success': True,
                'answer': answer,
                'response_time_seconds': elapsed_time,
                'num_source_docs': len(source_docs),
                'retrieval_k': retrieval_k,
                'prompt_type': prompt_type,
                'timestamp': start_time.isoformat(),
                'error': None
            }

        except Exception as e:
            end_time = datetime.now()
            elapsed_time = (end_time - start_time).total_seconds()

            return {
                'success': False,
                'answer': "",
                'response_time_seconds': elapsed_time,
                'num_source_docs': 0,
                'retrieval_k': retrieval_k,
                'prompt_type': prompt_type,
                'timestamp': start_time.isoformat(),
                'error': str(e)
            }

    def run_pipeline(self, csv_path: str, output_path: str, sample_size: Optional[int] = None,
                    retrieval_k: int = 15, prompt_type: str = "medical_rag") -> str:
        """
        Run the complete testing pipeline

        Args:
            csv_path: Path to input CSV with questions
            output_path: Path for output CSV with results
            sample_size: Number of questions to sample (None for all)
            retrieval_k: Number of documents to retrieve
            prompt_type: Type of prompt to use

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

        print(f"Running RAG pipeline with k={retrieval_k}, prompt_type={prompt_type}")
        print(f"Session ID: {self.session_id}")

        for i, question_data in enumerate(questions, 1):
            print(f"Processing question {i}/{len(questions)}: {question_data['person_name']} Q{question_data['question_number']}")

            # Run the question through RAG
            rag_result = self.run_single_question(
                question_data,
                retrieval_k=retrieval_k,
                prompt_type=prompt_type
            )

            # Combine question data with results
            result_row = {
                'session_id': self.session_id,
                'person_name': question_data['person_name'],
                'experience_years': question_data['experience_years'],
                'question_number': question_data['question_number'],
                'question_text': question_data['question_text'],
                'answer': rag_result['answer'],
                'success': rag_result['success'],
                'response_time_seconds': rag_result['response_time_seconds'],
                'num_source_docs': rag_result['num_source_docs'],
                'retrieval_k': rag_result['retrieval_k'],
                'prompt_type': rag_result['prompt_type'],
                'timestamp': rag_result['timestamp'],
                'error': rag_result['error']
            }

            results.append(result_row)

            # Print status
            if rag_result['success']:
                print(f"  Success ({rag_result['response_time_seconds']:.2f}s)")
            else:
                print(f"  Error: {rag_result['error']}")

        # Save results to CSV
        print(f"Saving results to {output_path}...")
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False, encoding='utf-8')

        # Print summary
        successful = sum(1 for r in results if r['success'])
        avg_time = sum(r['response_time_seconds'] for r in results if r['success']) / max(successful, 1)

        print(f"\nPipeline complete!")
        print(f"Total questions: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")
        print(f"Average response time: {avg_time:.2f}s")
        print(f"Results saved to: {output_path}")

        return output_path


def main():
    """Main function for running the pipeline"""
    # Configuration
    INPUT_CSV = r"C:\Users\agish\Desktop\RAG-Therapy\OneDrive_1_5-18-2025\Thera-RAG\comparisons\VHA Guideline LLM Survey (Responses).csv"
    OUTPUT_DIR = r"C:\Users\agish\Desktop\RAG-Therapy\OneDrive_1_5-18-2025\Thera-RAG\tests\results"

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"rag_test_results_{timestamp}.csv")

    # Initialize pipeline
    pipeline = RAGTestPipeline()

    # Run pipeline - modify these parameters as needed
    pipeline.run_pipeline(
        csv_path=INPUT_CSV,
        output_path=output_file,
        sample_size=None,  # Set to a number for sampling, None for all questions
        retrieval_k=15,
        prompt_type="medical_rag"
    )


if __name__ == "__main__":
    main()