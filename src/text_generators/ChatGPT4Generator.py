import os
from typing import List, Dict, Optional, AsyncGenerator
from openai import AsyncOpenAI
from loguru import logger
from src.text_generators.interface import Generator, GPT4GeneratorError

class GPT4Generator(Generator):
    """Generator using OpenAI's GPT-4 API."""
    
    def __init__(self):
        super().__init__()
        self.name = "GPT4Generator"
        self.description = "Generator using OpenAI's GPT-4 model"
        self.requires_library = ["openai"]
        self.requires_env = ["OPENAI_API_KEY"]
        self.streamable = True
        self.model = os.getenv("OPENAI_MODEL", "gpt-4")
        self.context_window = 10000
        self.context = {}
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    async def generate(
        self,
        queries: List[str],
        context: Optional[List[str]] = None,
    ) -> str:
        """Generate a response using GPT-4."""
        self.set_context(context or [])
        messages = self.prepare_messages(queries, self.context)

        try:
            completion = await self.client.chat.completions.create(
                messages=messages,
                model=self.model
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in GPT4Generator.generate: {e}")
            raise GPT4GeneratorError(f"Generation failed: {e}")

    async def generate_stream(
        self,
        queries: List[str],
        context: Optional[List[str]] = None
    ) -> AsyncGenerator[Dict[str, str], None]:
        """Generate a streaming response using GPT-4."""
        self.set_context(context or [])
        messages = self.prepare_messages(queries, self.context)

        try:
            stream = await self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                stream=True
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield {"content": chunk.choices[0].delta.content}

        except Exception as e:
            logger.error(f"Error in GPT4Generator.generate_stream: {e}")
            raise GPT4GeneratorError(f"Stream generation failed: {e}")

    def set_context(self, context: List[str]) -> None:
        """Set the context for generation."""
        self.context = {"context": context} if context else {}

    def prepare_messages(self, queries: List[str], context: Dict[str, List[str]]) -> List[Dict[str, str]]:
        """Prepare messages for the API call."""
        messages = []
        
        # Add system message if template is set
        if self.template:
            messages.append({
                "role": "system",
                "content": self.template.create_system_prompt()
            })
        
        # Add context if available
        if context.get("context"):
            messages.append({
                "role": "system",
                "content": "\n".join(context["context"])
            })
        
        # Add user queries
        for query in queries:
            messages.append({
                "role": "user",
                "content": query
            })
        
        return messages
