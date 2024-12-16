from src.text_generators.manager import get_text_generator_manager
from src.templates.handler import get_template_manager
from src.config import load_config
from src.db_setup import setup_database
from loguru import logger
import asyncio
import click

async def initialize_manager():
    """Initialize the text generator manager with configuration."""
    try:
        # Load configuration
        config = load_config()
        
        # Get managers
        text_generator_manager = get_text_generator_manager()
        template_manager = get_template_manager()
        
        # Set up default generator and template
        text_generator_manager.set_current_text_generator("gpt4")
        template_manager.set_current_template("coding")
        
        # Configure current generator
        current_generator = text_generator_manager.get_current_text_generator()
        current_generator.set_template(template_manager.get_current_template())
        
        # Initialize database
        await setup_database(config)
        
        return text_generator_manager
        
    except Exception as e:
        logger.error(f"Failed to initialize Hexamerous: {e}")
        raise

@click.group()
def cli():
    """Hexamerous CLI tool for managing code generation and documentation."""
    pass

@cli.command()
@click.option('--config', '-c', default=None, help='Path to configuration file')
def start(config):
    """Start the Hexamerous application."""
    try:
        asyncio.run(main(config))
    except KeyboardInterrupt:
        logger.info("Shutting down Hexamerous...")
    except Exception as e:
        logger.error(f"Error running Hexamerous: {e}")
        raise

async def main(config_path=None):
    """Main entry point for the Hexamerous application."""
    logger.info("Starting Hexamerous...")
    
    try:
        text_generator_manager = await initialize_manager()
        logger.info("Hexamerous initialized successfully")
        
        # Start the application loop
        while True:
            # Add your main application logic here
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        raise
    finally:
        logger.info("Shutting down Hexamerous...")

if __name__ == "__main__":
    cli()
