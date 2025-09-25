"""
Base prompt management classes
"""

from pathlib import Path
from typing import Dict, List, Optional
from langchain_core.prompts import PromptTemplate


class PromptManager:
    """Base class for managing prompt templates"""

    def __init__(self, templates_dir: Optional[Path] = None):
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "templates"

        self.templates_dir = Path(templates_dir)
        self.templates: Dict[str, str] = {}
        self._load_templates()

    def _load_templates(self):
        """Load all template files from the templates directory"""
        if not self.templates_dir.exists():
            self.templates_dir.mkdir(parents=True, exist_ok=True)
            return

        for template_file in self.templates_dir.glob("*.txt"):
            template_name = template_file.stem
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    self.templates[template_name] = f.read().strip()
            except Exception as e:
                print(f"Warning: Could not load template {template_name}: {e}")

    def get_prompt(self, template_name: str, **kwargs) -> PromptTemplate:
        """
        Get a formatted prompt template

        Args:
            template_name: Name of the template to use
            **kwargs: Variables to substitute in the template

        Returns:
            PromptTemplate instance
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found. Available: {list(self.templates.keys())}")

        template_str = self.templates[template_name]

        # Extract variables from template
        import re
        variables = re.findall(r'\{(\w+)\}', template_str)

        return PromptTemplate(
            template=template_str,
            input_variables=variables
        )

    def register_template(self, name: str, template: str):
        """
        Register a new prompt template

        Args:
            name: Template name
            template: Template string with {variable} placeholders
        """
        self.templates[name] = template

    def list_available_templates(self) -> List[str]:
        """List all available template names"""
        return list(self.templates.keys())

    def get_template_string(self, template_name: str) -> str:
        """Get the raw template string"""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        return self.templates[template_name]