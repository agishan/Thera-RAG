"""
Medical-specific prompt management
"""

from .base_prompts import PromptManager
from langchain_core.prompts import PromptTemplate


class MedicalPromptManager(PromptManager):
    """Specialized prompt manager for medical applications"""

    def __init__(self, templates_dir=None):
        super().__init__(templates_dir)
        self._register_default_templates()

    def _register_default_templates(self):
        """Register medical prompt templates"""

        # Single RAG template for medical Q&A with context
        self.register_template(
            "medical_rag",
            """You are a knowledgeable medical assistant with expertise in clinical guidelines and viscoelastic testing.

Use the following pieces of context to answer the question at the end. Provide concise, evidence-based answers to questions about therapy approaches and clinical practice. Be accurate and precise. If you don't know the answer based on the provided context, just say that you don't know.

{context}

Question: {question}
Answer:"""
        )

    def get_rag_prompt(self) -> PromptTemplate:
        """Get RAG prompt for medical Q&A with context"""
        return self.get_prompt("medical_rag")

    def get_vanilla_prompt(self) -> PromptTemplate:
        """Get vanilla prompt for medical Q&A without context"""
        return self.get_prompt("vanilla_medical")

    def get_cleansing_prompt(self) -> PromptTemplate:
        """Get cleansing prompt for bias removal"""
        return self.get_prompt("cleansing_medical")

    # Backward compatibility
    def get_clinical_qa_prompt(self, context_type="general") -> PromptTemplate:
        """Get clinical Q&A prompt (backward compatibility)"""
        return self.get_rag_prompt()