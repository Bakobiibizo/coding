from typing import List, Dict, Optional, AsyncGenerator
from abc import ABC, abstractmethod
from src.templates.interface import BaseTemplate

class GeneratorError(Exception):
    """Base class for all Generator errors."""
    pass

class GPT4GeneratorError(GeneratorError):
    """Error raised when an error occurs in the GPT4Generator."""
    pass

class Generator(ABC):
    """Abstract base class for text generators."""
    
    def __init__(self):
        self.name: str = ""
        self.description: str = ""
        self.requires_library: List[str] = []
        self.requires_env: List[str] = []
        self.streamable: bool = False
        self.model: str = ""
        self.context_window: int = 0
        self.context: Dict = {}
        self.api_key: Optional[str] = None
        self.base_url: Optional[str] = None
        self.template: Optional[BaseTemplate] = None

    @abstractmethod
    async def generate(
        self,
        queries: List[str],
        context: Optional[List[str]] = None
    ) -> str:
        """Generate text based on queries and context."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        queries: List[str],
        context: Optional[List[str]] = None
    ) -> AsyncGenerator[Dict[str, str], None]:
        """Generate streaming text based on queries and context."""
        pass

    def set_template(self, template: BaseTemplate) -> None:
        """Set the template for the generator."""
        self.template = template

    def get_template(self) -> Optional[BaseTemplate]:
        """Get the current template."""
        return self.template

class AvailableGenerators:
    """Registry of available text generators."""
    
    def __init__(self):
        self._generators: Dict[str, Generator] = {}

    def add_generator(self, name: str, generator: Generator) -> None:
        """Add a generator to the registry."""
        self._generators[name] = generator

    def remove_generator(self, name: str) -> None:
        """Remove a generator from the registry."""
        if name in self._generators:
            del self._generators[name]

    def get_generator(self, name: str) -> Generator:
        """Get a generator by name."""
        if name not in self._generators:
            raise KeyError(f"Generator '{name}' not found")
        return self._generators[name]

    def list_generators(self) -> Dict[str, Generator]:
        """List all available generators."""
        return self._generators.copy()

# Global registry of generators
available_generators = AvailableGenerators()
