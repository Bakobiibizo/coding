from typing import Dict, Optional
from src.text_generators.interface import Generator
from src.text_generators.ChatGPT4Generator import GPT4Generator
from src.text_generators.AgentArtificialGenerator import AgentArtificialGenerator

class TextGeneratorManager:
    """Manages text generators."""
    
    def __init__(self):
        """Initialize text generator manager with available generators."""
        self._text_generators = {
            "gpt4": GPT4Generator(),
            "agent": AgentArtificialGenerator(),
            # Add other generators here
        }
        self._current_text_generator: Optional[Generator] = None
        self.accumulated_tokens = 0
    
    def get_text_generator(self, text_generator_name: str) -> Generator:
        """Get a specific text generator by name."""
        if text_generator_name not in self._text_generators:
            raise ValueError(f"Text generator '{text_generator_name}' not found")
        return self._text_generators[text_generator_name]
    
    def set_current_text_generator(self, text_generator_name: str) -> None:
        """Set the current active text generator."""
        self._current_text_generator = self.get_text_generator(text_generator_name)
    
    def get_current_text_generator(self) -> Optional[Generator]:
        """Get the currently active text generator."""
        return self._current_text_generator
    
    def list_text_generators(self) -> Dict[str, Generator]:
        """List all available text generators."""
        return self._text_generators.copy()
    
    def update_token_count(self, tokens: int) -> None:
        """Update the accumulated token count."""
        self.accumulated_tokens += tokens
    
    def reset_token_count(self) -> None:
        """Reset the accumulated token count."""
        self.accumulated_tokens = 0

# Global text generator manager instance
_text_generator_manager = None

def get_text_generator_manager() -> TextGeneratorManager:
    """Get or create the global text generator manager instance."""
    global _text_generator_manager
    if _text_generator_manager is None:
        _text_generator_manager = TextGeneratorManager()
    return _text_generator_manager
