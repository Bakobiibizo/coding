# src/cli.py
import click
from loguru import logger
import os
from typing import Optional
from .config import load_config
from .db_setup import setup_database
import asyncio

@click.group()
def cli():
    """Hexamerous - AI-powered coding assistant and documentation tool."""
    pass

@cli.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('--output', '-o', default='project_docs.md', help='Output file for documentation')
@click.option('--format', '-f', type=click.Choice(['markdown', 'json']), default='markdown',
              help='Output format')
def document(directory: str, output: str, format: str):
    """Generate documentation for a project directory."""
    from .docs.walkdir import DirectoryMapper
    
    try:
        mapper = DirectoryMapper(
            max_depth=10,
            include_content=True,
            content_extensions=['.py', '.md', '.txt']
        )
        
        dir_map = mapper.map_directory(directory)
        mapper.save_map(dir_map, output, format)
        logger.info(f"Documentation saved to {output}")
        
    except Exception as e:
        logger.error(f"Error generating documentation: {e}")
        raise

@cli.command()
@click.argument('query')
@click.option('--model', '-m', default=None, help='Model to use for generation')
@click.option('--max-tokens', '-t', default=None, type=int, help='Maximum tokens to generate')
async def generate(query: str, model: Optional[str], max_tokens: Optional[int]):
    """Generate code based on a natural language query."""
    try:
        config = load_config()
        if model:
            config['model'] = model
        if max_tokens:
            config['max_tokens'] = max_tokens
            
        # Initialize database
        vector_db = await setup_database(config)
        
        # TODO: Implement generation logic
        logger.info("Code generation not yet implemented")
        
    except Exception as e:
        logger.error(f"Error in code generation: {e}")
        raise

@cli.command()
@click.argument('url')
@click.option('--output', '-o', default='weaviate_docs.md', help='Output file for documentation')
def collect_docs(url: str, output: str):
    """Collect documentation from a Weaviate instance."""
    from .docs.walkdir import WeaviateDocCollector
    
    try:
        collector = WeaviateDocCollector(url)
        collector.collect_docs()
        collector.save_to_markdown(output)
        logger.info(f"Documentation saved to {output}")
        
    except Exception as e:
        logger.error(f"Error collecting documentation: {e}")
        raise

def main():
    """Entry point for the CLI application."""
    try:
        cli()
    except Exception as e:
        logger.error(f"CLI error: {e}")
        raise

if __name__ == "__main__":
    main()