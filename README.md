# Hexamerous

## Overview

Hexamerous is a comprehensive coding assistant designed to facilitate development in Python and TypeScript. It offers a range of features, including context-aware code generation, vectorstore integration for long-term memory, and advanced UI components for managing interactions.

## Current Status

The project is currently in active development. Recent updates include the modularization of UI components, enhancements to text generation capabilities, and improvements to vectorstore integration. The following areas are still in progress:

- Completing the implementation of placeholder methods in UI components.
- Finalizing API interactions for text generators.
- Enhancing vectorstore functionality for efficient data retrieval and storage.

## Features

- **Contextual Code Generation**: Provides intelligent code suggestions based on the current context.
- **Vectorstore Integration**: Supports long-term memory and efficient document search.
- **UI Components**: Includes customizable widgets for managing user interactions.

## Installation

To set up the development environment using Poetry, follow these steps:

1. Ensure you have Python 3.8 or higher installed on your system.
2. Install Poetry by following the [official installation guide](https://python-poetry.org/docs/#installation).
3. Clone the repository and navigate to the project directory:

   ```bash
   git clone <repository-url>
   cd hexamerous
   ```

4. Install the project dependencies:

   ```bash
   poetry install
   ```

5. Run the application:

   ```bash
   poetry run python main.py
   ```

This will set up a virtual environment and install all necessary dependencies specified in the `pyproject.toml` file.

## Environment Variables

Configure the `.env` file with the necessary API keys and settings. Refer to the `env-example.env` file for guidance.

## Component Descriptions

### UI Components
- **ChatWidget**: Manages chat interactions and user input.
- **CustomTitleBar**: Provides a customizable title bar for the application.
- **LargeTextInputDialog**: Handles large text input from users.

### Text Generators
- **AgentArtificialGenerator**: Interfaces with the AgentArtificial API for text generation.

### Vectorstore
- **VectorDB**: Manages connections and operations with the vector database.
- **WeaviateManager**: Handles embedding and importing data into Weaviate.

## Contribution Guidelines

Contributions are welcome! To contribute:

1. Fork the repository and create a new branch for your feature or bugfix.
2. Ensure your code follows the project's coding standards and includes appropriate documentation.
3. Submit a pull request with a clear description of your changes.

## Future Plans

- Implementing additional features for text analysis and processing.
- Expanding support for more APIs and data sources.
- Improving the user interface for better usability and accessibility.

## Author

Bakobiibizo - richard@bakobi.com
Bakobi Inc. - https://sites.google.com/bakobi.com/bakobi-creative-design/home
