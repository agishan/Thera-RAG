"""
Prompt management system for RAG applications
"""

from .base_prompts import PromptManager
from .medical_prompts import MedicalPromptManager

__all__ = ['PromptManager', 'MedicalPromptManager']