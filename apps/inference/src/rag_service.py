from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List, Optional
import logging
from datetime import datetime
import time
import streamlit as st

# Import components (absolute imports for direct streamlit execution)
from .enhanced_retriever import EnhancedPineconeRetriever
from .prompts import MedicalPromptManager

class RAGService:
    """Enhanced RAG service with dynamic prompts and efficient retrieval"""

    def __init__(self, config):
        self.config = config
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.chain = None

        # API call tracking
        self.api_call_count = 0
        self.last_api_call = None

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize prompt manager
        self.prompt_manager = MedicalPromptManager()
        self.current_prompt_type = "medical_rag"

        # Get RAG prompt
        self.custom_prompt = self.prompt_manager.get_rag_prompt()

        self._setup_chain()
    
    def _setup_chain(self):
        """Set up the conversational retrieval chain with enhanced components"""
        # Initialize Pinecone (single instance)
        self.pc = Pinecone(api_key=self.config['pinecone_api_key'])
        self.index = self.pc.Index(self.config['pinecone_index_name'])
        self.embedder = SentenceTransformer(self.config['embedding_model'])

        # Create enhanced retriever with better reference management (re-ranking disabled)
        self.retriever = EnhancedPineconeRetriever(
            index=self.index,
            embedder=self.embedder,
            namespace=self.config['pinecone_namespace'],
            target_k=self.config['retrieval_k'],
            enable_reranking=False,  # Disabled for performance
            source_diversity_weight=0.2
        )
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.config['llm_model'],
            temperature=self.config['llm_temperature'],
            max_tokens=self.config['llm_max_tokens'],
            google_api_key=self.config['google_api_key']
        )

        # Create chain with custom prompt
        self._create_chain()

    def _create_chain(self, prompt_template=None):
        """Create or recreate the chain with specified prompt"""
        if prompt_template is None:
            prompt_template = self.custom_prompt

        self.chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=self.retriever,
            return_source_documents=True,
            verbose=False,
            combine_docs_chain_kwargs={"prompt": prompt_template}
        )
    
    def get_response(self, question, chat_history, retrieval_k=None, prompt_type=None):
        """
        Get response from the RAG chain with dynamic configuration

        Args:
            question: The user's question
            chat_history: List of (question, answer) tuples for conversation context
            retrieval_k: Number of documents to retrieve (optional)
            prompt_type: Type of prompt to use (optional)

        Returns:
            Response dictionary with answer and source documents
        """
        # Track API call timing
        current_time = datetime.now()
        self.api_call_count += 1

        # Calculate time since last call
        time_since_last = None
        if self.last_api_call:
            time_since_last = (current_time - self.last_api_call).total_seconds()

        # TERMINAL: Detailed logging for developer
        print(f"\n*** GEMINI API CALL #{self.api_call_count} ***")
        print(f"TIME: {current_time.strftime('%H:%M:%S.%f')[:-3]}")
        if time_since_last:
            print(f"TIME SINCE LAST CALL: {time_since_last:.2f} seconds")
        print(f"QUESTION: {question[:100]}{'...' if len(question) > 100 else ''}")
        print(f"MODEL: {self.config['llm_model']}")
        print(f"RETRIEVAL K: {retrieval_k or self.config['retrieval_k']}")
        print("CALLING GEMINI API NOW...")

        # BROWSER: User-friendly status
        try:
            with st.container():
                st.info(f"Calling Gemini API (Call #{self.api_call_count})")
                if time_since_last and time_since_last < 60:
                    st.caption(f"Time since last call: {time_since_last:.1f} seconds")
        except:
            pass  # In case streamlit context not available

        # Update retriever k value if provided (much more efficient now)
        if retrieval_k is not None and retrieval_k != self.retriever.k:
            self.retriever.update_k(retrieval_k)

        # Update prompt if different type requested
        if prompt_type is not None and prompt_type != self.current_prompt_type:
            self.set_prompt_type(prompt_type)

        # Time the API call
        start_time = time.time()

        try:
            # Get response from chain
            response = self.chain.invoke({
                "question": question,
                "chat_history": chat_history,
            })

            end_time = time.time()
            call_duration = end_time - start_time

            # TERMINAL: Detailed success logging
            print(f"SUCCESS! API call completed in {call_duration:.2f} seconds")
            print(f"Retrieved {len(response.get('source_documents', []))} documents")
            print(f"Response length: {len(response.get('answer', ''))} characters")

            # BROWSER: User-friendly success message
            try:
                st.success(f"API call successful ({call_duration:.1f}s)")
            except:
                pass

        except Exception as e:
            end_time = time.time()
            call_duration = end_time - start_time

            # TERMINAL: Detailed error logging
            print(f"ERROR: GEMINI API FAILED after {call_duration:.2f} seconds:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)[:200]}{'...' if len(str(e)) > 200 else ''}")

            # BROWSER: User-friendly error handling
            try:
                # Check for rate limit specifically
                if "429" in str(e) or "quota" in str(e).lower():
                    print(f"RATE LIMIT HIT! This is call #{self.api_call_count}")
                    if time_since_last:
                        print(f"WARNING: Only {time_since_last:.2f} seconds since last call!")

                    # Extract retry time from error message
                    import re
                    retry_match = re.search(r'retry in (\d+\.?\d*)s', str(e))
                    if retry_match:
                        retry_seconds = float(retry_match.group(1))
                        st.error(f"Rate limit exceeded. Please wait {retry_seconds:.0f} seconds before trying again.")
                        st.info(f"Gemini 2.5-Pro free tier: 2 requests per minute limit")
                    else:
                        st.error("Rate limit exceeded. Please wait 60 seconds before trying again.")
                else:
                    # Other API errors
                    st.error(f"API Error: {type(e).__name__}")
                    st.caption(f"Call duration: {call_duration:.1f} seconds")
            except:
                pass

            raise  # Re-raise the error

        finally:
            self.last_api_call = current_time
            print("-" * 60)

        return response

    def set_prompt_type(self, prompt_type: str):
        """
        Change the prompt type dynamically

        Args:
            prompt_type: Currently only 'medical_rag' is supported
        """
        try:
            if prompt_type in ["clinical_qa", "medical_rag"]:
                new_prompt = self.prompt_manager.get_rag_prompt()
            else:
                # Try to get it directly from prompt manager
                new_prompt = self.prompt_manager.get_prompt(prompt_type)

            self.custom_prompt = new_prompt
            self.current_prompt_type = prompt_type
            self._create_chain(new_prompt)

        except Exception as e:
            print(f"Warning: Could not set prompt type '{prompt_type}': {e}")
            print(f"Available prompts: {self.prompt_manager.list_available_templates()}")

    def get_available_prompt_types(self) -> List[str]:
        """Get list of available prompt types"""
        return self.prompt_manager.list_available_templates()

    def get_current_config(self) -> dict:
        """Get current configuration for debugging"""
        config = {
            "retrieval_k": self.retriever.k,
            "prompt_type": self.current_prompt_type,
            "namespace": self.retriever.namespace,
            "available_prompts": self.get_available_prompt_types()
        }

        # Add enhanced retriever stats if available
        if hasattr(self.retriever, 'get_retrieval_stats'):
            config.update(self.retriever.get_retrieval_stats())

        return config
