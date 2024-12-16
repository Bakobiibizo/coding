import asyncio
from src.text_generators.AnthropicGenerator import get_anthropic_generator

async def main():
    generator = get_anthropic_generator()
    queries = ["Write a Python function to calculate the factorial of a number."]
    context = []

    response = await generator.generate(queries, context)
    print(response)

    print("\nStreaming response:")
    async for chunk in generator.generate_stream(queries, context):
        print(chunk["message"], end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())