from typing import Dict, Optional
from src.templates.interface import BaseTemplate
from src.templates.saved_templates.coding_template import CodingTemplate

class TemplateManager:
    """Manages templates for text generation."""
    
    def __init__(self):
        """Initialize template manager with available templates."""
        self._templates = {
            "coding": CodingTemplate(),
            # Add other templates here
        }
        self._current_template: Optional[BaseTemplate] = None
    
    def get_template(self, template_name: str) -> BaseTemplate:
        """Get a specific template by name."""
        if template_name not in self._templates:
            raise ValueError(f"Template '{template_name}' not found")
        return self._templates[template_name]
    
    def set_current_template(self, template_name: str) -> None:
        """Set the current active template."""
        self._current_template = self.get_template(template_name)
    
    def get_current_template(self) -> Optional[BaseTemplate]:
        """Get the currently active template."""
        return self._current_template
    
    def list_templates(self) -> Dict[str, BaseTemplate]:
        """List all available templates."""
        return self._templates.copy()

# Global template manager instance
_template_manager = None

def get_template_manager() -> TemplateManager:
    """Get or create the global template manager instance."""
    global _template_manager
    if _template_manager is None:
        _template_manager = TemplateManager()
    return _template_manager

class Handler:
    def __init__(self, selected_template: str=None):
        self.template_manager = get_template_manager()
        if not selected_template:
            self.cli_select_template()
        self.set_current_template(selected_template)
    
    def get_selected_template(self, selected_template: str) -> str:
        self.set_current_template(selected_template)
        return self.template_manager.get_current_template().name
    
    def get_system_prompt(self) -> Dict[str, str]:
        return self.template_manager.get_current_template().create_system_prompt()
    
    def cli_select_template(self):
        print("Available Templates:")
        for template in self.template_manager.list_templates().keys():
            print(template)
        selected_template = input("Enter the name of the template you want to use: ")
        self.set_current_template(selected_template)
    
    def set_current_template(self, template_name: str) -> None:
        self.template_manager.set_current_template(template_name)

def get_handler(selected_template: str):
    return Handler(selected_template)

if __name__ == "__main__":
    print(get_handler("coding_template").get_system_prompt())