# Directory Structure

- 📁 **src/**
  - 📄 **__init__.py**

    📄 *File Path*: `./src/__init__.py`

    ```
    
    ```

  - 📁 **actions/**
    - 📄 **final_answer.py**

      📄 *File Path*: `./src/actions/final_answer.py`

      ```
      import json
from typing import Optional
from openai.types.beta.threads import ThreadMessage
from openai.pagination import SyncCursorPage
from utils.coala import CoALA
from utils.ops_api_handler import create_message_runstep
from utils.tools import ActionItem, Actions, actions_to_map
from utils.openai_clients import litellm_client, assistants_client
from data_models import run
import os


class FinalAnswer:
    def __init__(
        self,
        run_id: str,
        thread_id: str,
        assistant_id: str,
        tool_items: dict[str, ActionItem],
        job_summary: str,
    ):
        self.role_instructions = f"""Your role is to provide a text response to the user according to the messages and the current working memory."""
        self.run_id = run_id
        self.thread_id = thread_id
        self.assistant_id = assistant_id
        self.tool_items = tool_items
        self.job_summary = job_summary

    def generate(
        self,
        messages: SyncCursorPage[ThreadMessage],
        runsteps: SyncCursorPage[run.RunStep],
        content: Optional[str] = None,
    ) -> run.RunStep:
        if not content:
            # Compose the prompt for the summarization task
            coala = CoALA(
                runsteps=runsteps,
                messages=messages,
                job_summary=self.job_summary,
                tools_map=self.tool_items,
            )
            prompt = coala.compose_prompt("final_answer")
            print("\nFINALANSWER COALA PROMPT:\n", prompt)

            generator_messages = [
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
            response = litellm_client.chat.completions.create(
                model=os.getenv("LITELLM_MODEL"),  # Replace with your model of choice
                messages=generator_messages,
                max_tokens=500,  # You may adjust the token limit as necessary
            )
            content = response.choices[0].message.content
            content = content.split("Final Answer: ", 1)[1]

        run_step = create_message_runstep(
            self.thread_id, self.run_id, self.assistant_id, content
        )
        print("Final answer content: ", content)
        return run_step

      ```

    - 📄 **retrieval.py**

      📄 *File Path*: `./src/actions/retrieval.py`

      ```
      import json
from typing import Optional
from openai.types.beta.threads import ThreadMessage
from openai.pagination import SyncCursorPage
from utils.weaviate_utils import retrieve_file_chunks
from utils.ops_api_handler import create_retrieval_runstep
from utils.tools import ActionItem, Actions, actions_to_map
from utils.openai_clients import litellm_client, assistants_client
from data_models import run
import os

# import coala
from utils.coala import CoALA


class Retrieval:
    def __init__(
        self,
        run_id: str,
        thread_id: str,
        assistant_id: str,
        tools_map: dict[str, ActionItem],
        job_summary: str,
    ):
        self.query_maker_instructions = f"""Your role is generate a query for semantic search to retrieve important according to current working memory and the available files.
Even if there is no relevant information in the working memory, you should still generate a query to retrieve the most relevant information from the available files.
Only respond with the query iteself NOTHING ELSE."""
        self.run_id = run_id
        self.thread_id = thread_id
        self.assistant_id = assistant_id
        self.tool_items = tools_map
        self.job_summary = job_summary
        self.coala = None
        self.assistant = None

    def generate(
        self,
        messages: SyncCursorPage[ThreadMessage],
        runsteps: SyncCursorPage[run.RunStep],
        content: Optional[str] = None,
    ) -> run.RunStep:
        # get relevant retrieval query
        self.coala = CoALA(
            runsteps=runsteps,
            messages=messages,
            job_summary=self.job_summary,
            tools_map=self.tool_items,
        )

        messages = [
            {
                "role": "user",
                "content": self.compose_query_system_prompt(),
            },
        ]
        response = litellm_client.chat.completions.create(
            model=os.getenv("LITELLM_MODEL"),  # Replace with your model of choice
            messages=messages,
            max_tokens=200,  # You may adjust the token limit as necessary
        )
        query = response.choices[0].message.content
        print("Retrieval query: ", query)
        # TODO: retrieve from db, and delete mock retrieval document
        retrieved_documents = retrieve_file_chunks(self.assistant.file_ids, query)

        run_step = create_retrieval_runstep(
            self.thread_id, self.run_id, self.assistant_id, retrieved_documents
        )
        return run_step

    def compose_file_list(
        self,
    ) -> str:
        assistant = assistants_client.beta.assistants.retrieve(
            assistant_id=self.assistant_id,
        )
        self.assistant = assistant
        files_names = []
        for file_id in assistant.file_ids:
            file = assistants_client.files.retrieve(file_id)
            files_names.append(f"- {file.filename}")
        return "\n".join(files_names)

    def compose_query_system_prompt(self) -> str:
        trace = self.coala.compose_trace()

        composed_instruction = f"""{self.query_maker_instructions}

The files currently available to you are:
{self.compose_file_list()}

Current working memory:
Question: {self.job_summary}
{trace}"""
        print("\n\nRETRIEVAL SYSTEM PROMP: ", composed_instruction)
        return composed_instruction

      ```

    - 📄 **text_generation.py**

      📄 *File Path*: `./src/actions/text_generation.py`

      ```
      import json
from typing import Optional
from openai.types.beta.threads import ThreadMessage
from openai.pagination import SyncCursorPage
from utils.coala import CoALA
from utils.ops_api_handler import create_message_runstep
from utils.tools import ActionItem, Actions, actions_to_map
from utils.openai_clients import litellm_client, assistants_client
from data_models import run
import os


class TextGeneration:
    def __init__(
        self,
        run_id: str,
        thread_id: str,
        assistant_id: str,
        tool_items: dict[str, ActionItem],
        job_summary: str,
    ):
        self.role_instructions = f"""Your role is to provide a text response to the user according to the messages and the current working memory."""
        self.run_id = run_id
        self.thread_id = thread_id
        self.assistant_id = assistant_id
        self.tool_items = tool_items
        self.job_summary = job_summary

    def generate(
        self,
        messages: SyncCursorPage[ThreadMessage],
        runsteps: SyncCursorPage[run.RunStep],
        content: Optional[str] = None,
    ) -> run.RunStep:
        if not content:
            # Compose the prompt for the summarization task
            coala = CoALA(
                runsteps=runsteps,
                messages=messages,
                job_summary=self.job_summary,
                tools_map=self.tool_items,
            )
            prompt = coala.compose_prompt("thought")
            print("\n\nTEXTGENERATOR COALA PROMPT:\n", prompt)

            generator_messages = [
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
            response = litellm_client.chat.completions.create(
                model=os.getenv("LITELLM_MODEL"),  # Replace with your model of choice
                messages=generator_messages,
                max_tokens=500,  # You may adjust the token limit as necessary
            )
            content = response.choices[0].message.content
            print("\n\nTEXTGENERATOR RESPONSE:\n", content)
            content = content.split("Thought:", 1)[1]
            print("\n\nTEXTGENERATOR RESPONSE SPLIT:\n", content)
            content = content.split("Action:", 1)[0]

        run_step = create_message_runstep(
            self.thread_id, self.run_id, self.assistant_id, content
        )
        print("TextGeneration content: ", content)
        return run_step

    def compose_working_memory(
        self,
    ) -> str:
        steps = assistants_client.beta.threads.runs.steps.list(
            thread_id=self.thread_id,
            run_id=self.run_id,
        )
        return "\n".join(
            [json.dumps(step.step_details.model_dump()) for step in steps.data]
        )

    def compose_system_prompt(self) -> str:
        working_memory = self.compose_working_memory()

        composed_instruction = f"""{self.role_instructions}

Current working memory:
{working_memory}"""
        print("\n\nTEXTGENERATION SYSTEM PROMP: ", composed_instruction)
        return composed_instruction

      ```

    - 📄 **web_retrieval.py**

      📄 *File Path*: `./src/actions/web_retrieval.py`

      ```
      import requests
from bs4 import BeautifulSoup
from typing import Optional
from openai.types.beta.threads import ThreadMessage
from openai.pagination import SyncCursorPage
from utils.weaviate_utils import retrieve_file_chunks
from utils.ops_api_handler import create_web_retrieval_runstep
from utils.tools import ActionItem, Actions, actions_to_map
from utils.openai_clients import litellm_client, assistants_client
from data_models import run
import os
from utils.coala import CoALA


# TODO: Sean, update this, this is where the execution happens
class WebRetrieval:
    def __init__(
        self,
        run_id: str,
        thread_id: str,
        assistant_id: str,
        tools_map: dict[str, ActionItem],
        job_summary: str,
    ):
        self.query_maker_instructions = f"""Your role is to generate a search query based on the current working memory and the job summary.
Only respond with the search query itself, NOTHING ELSE."""
        self.run_id = run_id
        self.thread_id = thread_id
        self.assistant_id = assistant_id
        self.tool_items = tools_map
        self.job_summary = job_summary
        self.coala = None
        self.assistant = None

    def generate(
        self,
        messages: SyncCursorPage[ThreadMessage],
        runsteps: SyncCursorPage[run.RunStep],
        content: Optional[str] = None,
    ) -> run.RunStep:
        self.coala = CoALA(
            runsteps=runsteps,
            messages=messages,
            job_summary=self.job_summary,
            tools_map=self.tool_items,
        )

        query = self.generate_search_query()
        print("Web retrieval query:", query)

        search_results = self.perform_web_search(query)
        
        run_step = create_web_retrieval_runstep(
            self.thread_id,
            self.run_id,
            self.assistant_id,
            search_results,
            site="google.com",  # You might want to make this configurable
        )
        return run_step

    def generate_search_query(self) -> str:
        messages = [
            {
                "role": "user",
                "content": self.compose_query_system_prompt(),
            },
        ]
        response = litellm_client.chat.completions.create(
            model=os.getenv("LITELLM_MODEL"),  # Replace with your model of choice
            messages=messages,
            max_tokens=200,  # You may adjust the token limit as necessary
        )
        return response.choices[0].message.content

    def perform_web_search(self, query: str) -> str:
        # This is a simple implementation. You might want to use a proper search API for production.
        url = f"https://www.google.com/search?q={query}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract search results
        search_results = []
        for g in soup.find_all('div', class_='g'):
            anchor = g.find('a')
            if anchor:
                link = anchor['href']
                title = anchor.find('h3')
                if title:
                    title = title.text
                    search_results.append(f"{title}\n{link}\n\n")
        
        return "".join(search_results[:5])  # Return top 5 results

    def compose_query_system_prompt(self) -> str:
        trace = self.coala.compose_trace()

        composed_instruction = f"""{self.query_maker_instructions} 

Current working memory:
Question: {self.job_summary}
{trace}"""  # TODO: Sean, this prompt should not change too much
        print("\n\nWEB RETRIEVAL SYSTEM PROMP: ", composed_instruction)
        return composed_instruction

      ```

  - 📁 **agents/**
    - 📄 **orchestrator.py**

      📄 *File Path*: `./src/agents/orchestrator.py`

      ```
      from openai.types.beta.threads import ThreadMessage
from data_models import run
from openai.pagination import SyncCursorPage
from utils.tools import ActionItem, Actions, actions_to_map
from utils.openai_clients import litellm_client, assistants_client
import os
from utils.coala import CoALA


class OrchestratorAgent:
    def __init__(
        self,
        run_id: str,
        thread_id: str,
        tool_items: dict[str, ActionItem],
        job_summary: str,
    ):
        self.run_id = run_id
        self.thread_id = thread_id
        self.tool_items = tool_items
        self.action_items = actions_to_map(
            [Actions.TEXT_GENERATION.value, Actions.COMPLETION.value]
        )
        self.job_summary = job_summary

    def generate(
        self,
        messages: SyncCursorPage[ThreadMessage],
        runsteps: SyncCursorPage[run.RunStep],
    ) -> Actions:
        """
        Generate a summary of the chat history with a focus on the current user request and tool usage.

        Args:
            messages (SyncCursorPage[Message]): The chat messages.
            runsteps (SyncCursorPage[run.RunStep]): The run steps.

        Returns:
            Actions: The action to be taken based on the generated summary.
        """
        # Compose the prompt for the summarization task
        coala = CoALA(
            runsteps=runsteps,
            messages=messages,
            job_summary=self.job_summary,
            tools_map=self.tool_items,
        )
        prompt = coala.compose_prompt("action")
        print("\n\nORCHESTRATOR COALA PROMPT:\n", prompt)

        generator_messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ]

        # Call to the AI model to generate the summary
        response = litellm_client.chat.completions.create(
            model=os.getenv("LITELLM_MODEL"),  # Replace with your model of choice
            messages=generator_messages,
            max_tokens=100,  # You may adjust the token limit as necessary
        )
        content = response.choices[0].message.content
        content = content.replace("\\", "")
        print("ORCHESTRATOR GENERATION: ", response.choices[0].message.content)
        for key in self.tool_items.keys():
            if f"{key}" in content:
                print("KEY: ", f"<{key}>")
                return Actions(key)
        for key in self.action_items.keys():
            if f"{key}" in content:
                print("KEY: ", f"<{key}>")
                return Actions(key)

        return Actions.FAILURE

      ```

    - 📄 **router.py**

      📄 *File Path*: `./src/agents/router.py`

      ```
      from openai.types.beta.threads import ThreadMessage
from openai.pagination import SyncCursorPage
from constants import PromptKeys
from utils.tools import ActionItem
from utils.openai_clients import litellm_client
import os


class RouterAgent:
    def __init__(
        self,
    ):
        self.role_instructions = f"""Your role is to determine whether to use tools or directly generate a response.
In the case that you think you may need more tools, simply respond with '{PromptKeys.TRANSITION.value}'. Otherwise, generate an appropriate response."""  # noqa

    def compose_system_prompt(self, tools: dict[str, ActionItem]) -> str:
        tools_list = "\n".join(
            [f"- {tool.type}: {tool.description}" for _, tool in tools.items()]
        )
        return f"""{self.role_instructions}

The tools available to you are:
{tools_list}"""

    # TODO: add assistant and base tools off of assistant
    def generate(
        self,
        tools: dict[str, ActionItem],
        paginated_messages: SyncCursorPage[ThreadMessage],
    ) -> str:
        """
        Generates a response based on the chat history and role instructions.

        Args:
            tools (dict): The tools available to the agent.
            paginated_messages (SyncCursorPage[Message]): The chat history.

        Returns:
            str: It either returns `{PromptKeys.TRANSITION.value}` or a generated response.
        """

        messages = [
            {
                "role": "system",
                "content": self.compose_system_prompt(tools),
            }
        ]
        print("\n\nSYSTEM PROMPT: ", messages[0]["content"])
        for message in paginated_messages.data:
            messages.append(
                {
                    "role": message.role,
                    "content": message.content[0].text.value,
                }
            )
        print("MESSAGES: ", messages)
        response = litellm_client.chat.completions.create(
            model=os.getenv("LITELLM_MODEL"),
            messages=messages,
            max_tokens=500,
        )

        print("GENERATION: ", response.choices[0].message.content)
        if PromptKeys.TRANSITION.value in response.choices[0].message.content:
            return PromptKeys.TRANSITION.value
        else:
            return response.choices[0].message.content

      ```

    - 📄 **summarizer.py**

      📄 *File Path*: `./src/agents/summarizer.py`

      ```
      from openai.types.beta.threads import ThreadMessage
from openai.pagination import SyncCursorPage
from utils.tools import ActionItem
from utils.openai_clients import litellm_client
import os


class SummarizerAgent:
    def __init__(self):
        pass

    def generate(
        self,
        tools: dict[str, ActionItem],
        paginated_messages: SyncCursorPage[ThreadMessage],
    ) -> str:
        """
        Create a summary of the chat history with an emphasis on the current user request and tool use.

        Args:
            tools (dict): A dictionary containing available tools and their descriptions.
            chat_history (list): A list of messages representing the chat history.

        Returns:
            str: A summary useful for planning and tool use.
        """
        messages = []
        for message in paginated_messages.data:
            messages.append(
                {
                    "role": message.role,
                    "content": message.content[0].text.value,
                }
            )

        # pass the content of the last message to the compose_prompt method and replace the latest_message variable with the content of the last message
        latest_message = messages[-1]["content"]
        modified_prompt = self.compose_prompt(tools, latest_message)
        messages[-1]["content"] = modified_prompt

        # Call to the AI model to generate the summary
        response = litellm_client.chat.completions.create(
            model=os.getenv("LITELLM_MODEL"),
            messages=messages,
            max_tokens=1000,  # You may adjust the token limit as necessary
        )

        # Extract the summary from the response
        summary = response.choices[0].message.content
        return summary

    def compose_prompt(self, tools: dict[str, ActionItem], latest_message: str) -> str:
        tools_list = "\n".join(
            [f"- {tool.type}: {tool.description}" for _, tool in tools.items()]
        )
        return f"""Summarize the purpose of the message <<{latest_message}>> into a single comprehensive statement.
Ensure that the summary includes all relevant details needed for effective use of the following tools:
{tools_list}"""

      ```

  - 📄 **constants.py**

    📄 *File Path*: `./src/constants.py`

    ```
    from enum import Enum


class PromptKeys(Enum):
    TRANSITION = "<TRANSITION>"

    ```

  - 📄 **consumer.py**

    📄 *File Path*: `./src/consumer.py`

    ```
    import threading
from concurrent.futures import ThreadPoolExecutor
import pika
import os
from dotenv import load_dotenv
from run_executor.main import ExecuteRun
import json

load_dotenv()

RABBITMQ_USER = os.getenv("RABBITMQ_USER")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST")
RABBITMQ_PORT = os.getenv("RABBITMQ_PORT")


class RabbitMQConsumer:
    def __init__(
        self, max_workers=5
    ):  # max_workers can be adjusted based on demand # noqa
        credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=RABBITMQ_HOST,
                port=RABBITMQ_PORT,
                credentials=credentials,
                heartbeat=30,
            )
        )
        self.channel = self.connection.channel()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def process_message(self, body):
        message = body.decode("utf-8")
        data = json.loads(message)

        print(f"Processing {data}")
        run = ExecuteRun(data["thread_id"], data["run_id"])
        run.execute()
        # Insert your Run Executor pipeline logic here

    def callback(self, ch, method, properties, body):
        try:
            self.executor.submit(self.process_message_and_ack, body, ch, method)
        except Exception as e:
            print(f"Failed to submit the task to the executor: {e}")

    def process_message_and_ack(self, body, ch, method):
        try:
            self.process_message(body)
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            print(f"Failed to process message {body}: {e}")
            # Here you can decide whether to reject, requeue or just log the exception
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    def start_consuming(self, queue_name):
        self.channel.queue_declare(queue=queue_name, durable=True)
        self.channel.basic_consume(
            queue=queue_name, on_message_callback=self.callback, auto_ack=False
        )
        print("Waiting for messages. To exit press CTRL+C")
        self.channel.start_consuming()


consumer = RabbitMQConsumer()
consumer.start_consuming("runs_queue")

    ```

  - 📁 **data_ingestion/**
    - 📄 **data_source.py**

      📄 *File Path*: `./src/data_ingestion/data_source.py`

      ```
      from abc import ABC, abstractmethod
from typing import Any, List

class DataSource(ABC):
    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def fetch_data(self, query: Any) -> List[Any]:
        pass

    @abstractmethod
    def process_data(self, data: List[Any]) -> List[Any]:
        pass

class PDFDataSource(DataSource):
    def connect(self) -> bool:
        # Implement connection logic
        pass

    def fetch_data(self, query: Any) -> List[Any]:
        # Implement PDF data fetching logic
        pass

    def process_data(self, data: List[Any]) -> List[Any]:
        # Implement PDF data processing logic
        pass

class WebScraperDataSource(DataSource):
    def connect(self) -> bool:
        # Implement connection logic
        pass

    def fetch_data(self, query: Any) -> List[Any]:
        # Implement web scraping logic
        pass

    def process_data(self, data: List[Any]) -> List[Any]:
        # Implement web scraping data processing logic
        pass

# Add more data source classes as needed
      ```

    - 📄 **web_retrieval.py**

      📄 *File Path*: `./src/data_ingestion/web_retrieval.py`

      ```
      import requests
from bs4 import BeautifulSoup
from typing import Optional
from openai.types.beta.threads import ThreadMessage
from openai.pagination import SyncCursorPage
from utils.ops_api_handler import create_web_retrieval_runstep
from utils.tools import ActionItem
from utils.openai_clients import litellm_client
from data_models import run
import os
from utils.coala import CoALA

class WebRetrieval:
    def __init__(
        self,
        run_id: str,
        thread_id: str,
        assistant_id: str,
        tools_map: dict[str, ActionItem],
        job_summary: str,
    ):
        self.query_maker_instructions = f"""Your role is to generate a search query based on the current working memory and the job summary.
Only respond with the search query itself, NOTHING ELSE."""
        self.run_id = run_id
        self.thread_id = thread_id
        self.assistant_id = assistant_id
        self.tool_items = tools_map
        self.job_summary = job_summary
        self.coala = None

    def generate(
        self,
        messages: SyncCursorPage[ThreadMessage],
        runsteps: SyncCursorPage[run.RunStep],
        content: Optional[str] = None,
    ) -> run.RunStep:
        self.coala = CoALA(
            runsteps=runsteps,
            messages=messages,
            job_summary=self.job_summary,
            tools_map=self.tool_items,
        )

        query = self.generate_search_query()
        print("Web retrieval query:", query)

        search_results = self.perform_web_search(query)
        
        run_step = create_web_retrieval_runstep(
            self.thread_id,
            self.run_id,
            self.assistant_id,
            search_results,
            site="google.com",  # You might want to make this configurable
        )
        return run_step

    def generate_search_query(self) -> str:
        messages = [
            {
                "role": "user",
                "content": self.compose_query_system_prompt(),
            },
        ]
        response = litellm_client.chat.completions.create(
            model=os.getenv("LITELLM_MODEL"),
            messages=messages,
            max_tokens=200,
        )
        return response.choices[0].message.content

    def perform_web_search(self, query: str) -> str:
        # This is a simple implementation. You might want to use a proper search API for production.
        url = f"https://www.google.com/search?q={query}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract search results
        search_results = []
        for g in soup.find_all('div', class_='g'):
            anchor = g.find('a')
            if anchor:
                link = anchor['href']
                title = anchor.find('h3')
                if title:
                    title = title.text
                    search_results.append(f"{title}\n{link}\n\n")
        
        return "".join(search_results[:5])  # Return top 5 results

    def compose_query_system_prompt(self) -> str:
        trace = self.coala.compose_trace()

        composed_instruction = f"""{self.query_maker_instructions} 

Current working memory:
Question: {self.job_summary}
{trace}"""
        print("\n\nWEB RETRIEVAL SYSTEM PROMPT: ", composed_instruction)
        return composed_instruction
      ```

  - 📁 **data_models/**
    - 📄 **__init__.py**

      📄 *File Path*: `./src/data_models/__init__.py`

      ```
      
      ```

    - 📄 **chunk.py**

      📄 *File Path*: `./src/data_models/chunk.py`

      ```
      from pydantic import BaseModel, Field
from typing import Union, List, Dict
from datetime import date as Date
from uuid import uuid4
from src.data_models.component import Component

class Chunk(BaseModel):
    text: str = Field(default="")
    text_no_overlap: str = Field(default="")
    doc_name: str = Field(default="")
    doc_type: str = Field(default="")
    doc_uuid: str = Field(default="")
    chunk_id: str = Field(default="")

    def __setattr__(self, name, value):
       if name in self.__dict__:
           field = self.__fields__.get(name)
           if field:
               value = field.(value)
           super().__setattr__(name, value)
       else:
           raise AttributeError(f"Document has no attribute {name}")
       
    class Config:
        # If you need to handle arbitrary data types or custom validation
        # you can add configurations here
        pass

class Chunker(Component):
    """
    Interface for Verba Chunking.
    """

    def __init__(self):
        super().__init__()
        self.input_form = InputForm.CHUNKER.value  # Default for all Chunkers
        self.default_units = 100
        self.default_overlap = 50

    def chunk(
        self, documents: list[Document], units: int, overlap: int
    ) -> list[Document]:
        """Chunk verba documents into chunks based on units and overlap.

        @parameter: documents : list[Document] - List of Verba documents
        @parameter: units : int - How many units per chunk (words, sentences, etc.)
        @parameter: overlap : int - How much overlap between the chunks
        @returns list[str] - List of documents that contain the chunks.
        """
        raise NotImplementedError("chunk method must be implemented by a subclass.")

      ```

    - 📄 **component.py**

      📄 *File Path*: `./src/data_models/component.py`

      ```
      from pydantic import BaseModel
from typing import List, Dict, Optional, Union, Any


class Component(BaseModel):
    name: str
    required_env: List[str]
    required_library: List[str]
    description: str

      ```

    - 📄 **document.py**

      📄 *File Path*: `./src/data_models/document.py`

      ```
      from pydantic import BaseModel, Field
from typing import Union, List, Dict
from datetime import date as Date


class Document(BaseModel):
    text: str = Field(default="")
    type: str = Field(default="")
    name: str = Field(default="")
    path: str = Field(default="")
    link: str = Field(default="")
    timestamp: Date = Field(default=Date.today().ctime().format("%Y-%m-%d %H:%M:%S"))
    reader: str
    meta: Dict = Field(default={})
    doc_uuid: str = Field(default="")
    chunk_id: str = Field(default="")
    tokens: int = Field(default=0)
    vector: Union[List[float], None] = Field(default=None)
    score: float = Field(default=0.0)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            field = self.__fields__.get(name)
            if field:
                value = field.type_(value)
            super().__setattr__(name, value)
        else:
            raise AttributeError(f"Document has no attribute {name}")

    class Config:
        # If you need to handle arbitrary data types or custom validation
        # you can add configurations here
        pass

      ```

    - 📄 **generators.py**

      📄 *File Path*: `./src/data_models/generators.py`

      ```
      """Abstract base class for text generators."""
from pydantic import BaseModel
from typing import List, Dict, Iterator
from src.data_models.templates import BaseTemplate
from abc import ABC, abstractmethod


class Context(BaseModel):
    """Context for the text generator."""

    context: List[Dict[str, str]]
    context_window: int


class OAIMessage(BaseModel):
    """An OpenAI message."""

    role: str
    content: str


class OAIResponse(BaseModel):
    """An OpenAI response."""

    id: str
    object: str
    created: int
    model: str
    choices: List[OAIMessage]
    usage: Dict[str, int]


class OAIRequest(BaseModel):
    """An OpenAI request."""

    model: str
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    stop: List[str]
    stream: bool
    n: int


class GeneratorConfig(BaseModel):
    """The configuration for a text generator."""

    template: BaseTemplate
    url: str
    api: str
    context: List[Dict[str, str]]
    context_window: int


class Generator(BaseModel, ABC):
    """Abstract base class for text generators."""

    template: BaseTemplate
    url: str
    api_key: str
    context: List[Dict[str, str]]
    context_window: int

    def __init__(self, generator_config: "GeneratorConfig"):
        """
        Initializes the object with the given generator_config.

        Parameters
        ----------
        generator_config: GeneratorConfig
            The generator_config object containing the configuration values.
        """
        self.template = self.set_template(generator_config.template)
        self.url = self.set_template(generator_config.url)
        self.api_key = self.set_template(generator_config.api)
        self.context = self.set_template(generator_config.context)
        self.context_window = self.set_context_window(generator_config.context_window)

    def set_apikey(self, apikey) -> bool:
        """Set the API key for the text generation API.

        Parameters
        ----------
        apikey : str
            The API key for the text generation API.

        Returns
        -------
        bool
            `True` if API key was successfully set, `False` otherwise.
        """
        self.api_key = apikey
        return True

    def set_url(self, url) -> bool:
        """Set the URL of the text generation API.

        Parameters
        ----------
        url : str
            The URL of the text generation API.

        Returns
        -------
        bool
            `True` if URL was successfully set, `False` otherwise.
        """
        self.url = url
        return True

    def set_context_window(self, context_window: int) -> bool:
        """Set the maximum number of tokens to include in the context for the text generation request.
        Parameters
        ----------
        context_window : int
            The maximum number of tokens to include in the context for the text generation request.

        Returns
        -------
        bool
            `True` if context window was successfully set, `False` otherwise.
        """
        self.context_window = context_window
        return True

    def set_context(self, context: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Set the context for the text generation request.

        Parameters
        ----------
        context : List[Dict[str, str]]
            The context for the text generation request.

        Returns
        -------
        List[Dict[str, str]]
            The updated context for the text generation request.
        """
        if not self.context:
            self.context = []
            self.context.extend(iter(context))
        return self.context

    def set_template(self, template: BaseTemplate) -> List[Dict[str, str]]:
        """Set the template to use for generating text.

        Parameters
        ----------
        template : BaseTemplate
            The template to use for generating text.

        Returns
        -------
        List[Dict[str, str]]
            The updated context for the text generation request.
        """
        self.template = template
        if self.context[0]["role"] == "system":
            self.context.pop(0)
        self.context.insert(0, self.template.create_system_prompt())
        return self.context

    @abstractmethod
    async def generate(self, messages: List[Dict[str, str]]) -> Dict[str, str]:
        """Generate a response based on a list of input messages.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            The list of input messages.

        Returns
        -------
        Dict[str, str]
            The response generated by the text generator.
        """
        pass

    @abstractmethod
    async def async_generate(
        self, messages: List[Dict[str, str]]
    ) -> Iterator[Dict[str, str]]:
        """
        Asynchronously generate a response based on a list of input messages.
        """
        pass

    @abstractmethod
    def prepare_messages(
        self,
        queries: List[str],
        context: List[Dict[str, str]],
        messages: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        Prepares a list of messages for the text generator.
        """
        pass


class GeneratorConfig(BaseModel):
    """
    Configuration for the text generator.
    """

    template: BaseTemplate
    url: str
    api: str
    context: List
    context_window: int

      ```

    - 📄 **generics.py**

      📄 *File Path*: `./src/data_models/generics.py`

      ```
      from pydantic import BaseModel
from typing import Any, Dict


class DynamicManager(BaseModel):
    """
    A generic object that allows dynamic addition of attributes.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Dynamically set an attribute on the instance.

        Args:
            name (str): The name of the attribute.
            value (Any): The value of the attribute.
        """
        self.__dict__[name] = value

    def __getattr__(self, name: str) -> Any:
        """
        Retrieve the value of an attribute by its name.

        Args:
            name (str): The name of the attribute to retrieve.

        Returns:
            Any: The value of the attribute.

        Raises:
            AttributeError: If the attribute does not exist.
        """
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __delattr__(self, name: str) -> None:
        """
        Remove an attribute from the instance by its name.

        Args:
            name (str): The name of the attribute to remove.

        Raises:
            AttributeError: If the attribute does not exist.
        """
        if name in self.__dict__:
            del self.__dict__[name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def add_field(self, key: str, value: Any) -> None:
        """
        Add a new attribute to the template.

        Args:
            key (str): The name for the new attribute.
            value (Any): The value for the new attribute.
        """
        setattr(self, key, value)

    def get_field(self, key: str) -> Any:
        """
        Retrieve the value of an attribute by its name.

        Args:
            key (str): The name of the attribute to retrieve.

        Returns:
            Any: The value of the attribute.

        Raises:
            AttributeError: If the attribute does not exist.
        """
        return getattr(self, key)

    def remove_field(self, key: str) -> None:
        """
        Remove an attribute from the template by its name.

        Args:
            key (str): The name of the attribute to remove.

        Raises:
            AttributeError: If the attribute does not exist.
        """
        delattr(self, key)

    def update_field(self, key: str, value: Any) -> None:
        """
        Update the value of an existing attribute.

        Args:
            key (str): The name of the attribute to update.
            value (Any): The new value for the attribute.

        Raises:
            AttributeError: If the attribute does not exist.
        """
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise AttributeError(f"Attribute '{key}' does not exist in the template.")

    def get_all_fields(self) -> Dict[str, Any]:
        """
        Get all attributes in the template.

        Returns:
            Dict[str, Any]: A dictionary containing all attribute-value pairs in the template.
        """
        templates_dict = {}
        for key, value in self.dict().items():
            if isinstance(value, dict):
                templates_dict[key] = value
        return templates_dict

      ```

    - 📄 **input_form.py**

      📄 *File Path*: `./src/data_models/input_form.py`

      ```
      from enum import Enum
from src.data_models.component import Component
from src.data_models.document import Document


class InputForm(Enum):
    UPLOAD = "UPLOAD"  # Input Form to upload text files directly
    INPUT = "INPUT"  # Simple Text Input in Frontend
    CHUNKER = "CHUNKER"  # Default Input for Chunkers
    TEXT = "TEXT"  # Default Input for Embedder


class Reader(Component):
    """
    Interface for Verba Readers.
    """

    def __init__(self):
        super().__init__()
        self.file_types = []
        self.input_form = InputForm.UPLOAD.value

    def load(
        bytes: list[str],
        contents: list[str],
        paths: list[str],
        fileNames: list[str],
        document_type: str,
    ) -> list[Document]:
        """Ingest data into Weaviate
        @parameter: bytes : list[str] - List of bytes
        @parameter: contents : list[str] - List of string content
        @parameter: paths : list[str] - List of paths to files
        @parameter: fileNames : list[str] - List of file names
        @parameter: document_type : str - Document type
        @returns list[Document] - Lists of documents.
        """
        raise NotImplementedError("load method must be implemented by a subclass.")

      ```

    - 📄 **run.py**

      📄 *File Path*: `./src/data_models/run.py`

      ```
      from pydantic import BaseModel
from typing import Literal, Optional, List, Any, Union
from enum import Enum
from openai.types.beta.threads.run import Run
from openai.types.beta.threads.runs import (
    RunStep,
    MessageCreationStepDetails,
    ToolCallsStepDetails,
)

Run
RunStep


class RunUpdate(BaseModel):
    assistant_id: Optional[str] = None
    cancelled_at: Optional[int] = None
    completed_at: Optional[int] = None
    expires_at: Optional[int] = None
    failed_at: Optional[int] = None
    file_ids: Optional[List[str]] = None
    instructions: Optional[str] = None
    last_error: Optional[Any] = None
    model: Optional[str] = None
    started_at: Optional[int] = None
    status: Optional[str] = None
    tools: Optional[Any] = None
    usage: Optional[Any] = None


class RunStatus(str, Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    REQUIRES_ACTION = "requires_action"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    FAILED = "failed"
    COMPLETED = "completed"
    EXPIRED = "expired"


StepDetails = Union[MessageCreationStepDetails, ToolCallsStepDetails]


class RunStepCreate(BaseModel):
    # Define the fields required for creating a RunStep
    assistant_id: str
    step_details: Any
    type: Literal["message_creation", "tool_calls"]
    status: Literal["in_progress", "cancelled", "failed", "completed", "expired"]
    step_details: StepDetails

      ```

    - 📄 **runstep.py**

      📄 *File Path*: `./src/data_models/runstep.py`

      ```
      from pydantic import BaseModel
from typing import Literal, Optional, Dict, Any, Union
from openai.types.beta.threads.runs import (
    RunStep,
    MessageCreationStepDetails,
    ToolCallsStepDetails,
)

RunStep

StepDetails = Union[MessageCreationStepDetails, ToolCallsStepDetails]


class RunStepUpdate(BaseModel):
    assistant_id: Optional[str] = None
    cancelled_at: Optional[int] = None
    completed_at: Optional[int] = None
    expired_at: Optional[int] = None
    failed_at: Optional[int] = None
    last_error: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    status: Literal["in_progress", "cancelled", "failed", "completed", "expired"] = None
    step_details: StepDetails = None
    type: Literal["message_creation", "tool_calls"] = None
    usage: Optional[Dict[str, Any]] = None

      ```

    - 📄 **templates.py**

      📄 *File Path*: `./src/data_models/templates.py`

      ```
      from pydantic import BaseModel
from typing import List, Dict
from enum import Enum


class BaseTemplate(BaseModel):
    """
    A base template for creating structured templates with persona, task, example, tools, and system prompts.

    Attributes:
        description (str): The description of the template.
        persona (str): The persona associated with the template.
        task (str): The task described in the template.
        example (str): An example related to the task.
        tools (str): Tools used in the task.
        system_prompt (List[Dict[str, str]]): A list of system prompts with role and content.

    Methods:
        get_template() -> str:
            Returns a formatted template string with persona, task, example, and tools.

        update_templates(new_template: "BaseTemplate"):
            Updates the template attributes with a new template.

        create_system_prompt():
            Creates a system prompt based on the template content.
    """

    description: str
    persona: str
    task: str
    example: str
    tools: str
    system_prompt: List[Dict[str, str]]

    def get_template(self):
        """
        gets a formatted template string with persona, task, example, and tools
        """
        return f"""
Persona: {self.persona}

Task: {self.task}

Example: {self.example}

Tools: {self.tools}
"""

    def update_templates(self, new_template: "BaseTemplate"):
        """
        updates the template attributes with a new template
        """
        self.persona = new_template.persona
        self.task = new_template.task
        self.example = new_template.example
        self.tools = new_template.tools

    def create_system_prompt(self):
        """
        creates a system prompt based on the template content
        """
        if self.system_prompt is None:
            self.system_prompt = []
        self.system_prompt.append({"role": "system", "content": self.get_template()})
        return self.system_prompt


class AvailableTemplates(Enum):
    CODING: str = "coding"


class Templates(BaseModel):
    selected_template: BaseTemplate
    available_templates: AvailableTemplates
    templates: Dict[str, BaseTemplate]

      ```

    - 📄 **tokenizer.py**

      📄 *File Path*: `./src/data_models/tokenizer.py`

      ```
      from pydantic import BaseModel
from typing import Any, List, Dict, Optional, Union


class TokenizerConfig(BaseModel):
    name: str
    description: str
    model: str
    tokenizer: str
    device: Any
    model_path: str

      ```

    - 📄 **tools.py**

      📄 *File Path*: `./src/data_models/tools.py`

      ```
      from typing import List, Callable
from pydantic import BaseModel


class Tool(BaseModel):
    name: str
    function: Callable
    description: str


class Tools(BaseModel):
    def __init__(self, new_tools: List[Tool]):
        super().__init__()
        self.tools = self.set_tools(new_tools)

    def set_tools(self, new_tools: List[Tool]):
        self.tools = new_tools
        return self.tools

    def add(self, tool: Tool):
        self.tools.append(tool)

    def remove(self, tool: Tool):
        self.tools.remove(tool)

    def clear(self):
        self.tools = []

    def update(self, new_tools: List[Tool]):
        self.tools = new_tools

    def get(self, name: str):
        for tool in self.tools:
            if tool.name == name:
                return tool

    def use(self, name: str, query: str):
        for tool in self.tools:
            if tool.name == name:
                return tool.function(query)

    def list(self):
        returnString = ""
        for tool in self.tools:
            returnString += f"{tool.name}\n"
        return returnString

    def describe(self):
        returnString = ""
        for tool in self.tools:
            returnString += f"{tool.name}: {tool.description}\n"
        return returnString

    def __call__(self, query: str):
        for tool in self.tools:
            if tool.name in query:
                return tool.function(query)

      ```

  - 📁 **run_executor/**
    - 📄 **__init__.py**

      📄 *File Path*: `./src/run_executor/__init__.py`

      ```
      
      ```

    - 📄 **main.py**

      📄 *File Path*: `./src/run_executor/main.py`

      ```
      from typing import Dict, Any, Optional, List
from constants import PromptKeys
from utils.tools import ActionItem, Actions, tools_to_map
from utils.ops_api_handler import create_message_runstep, update_run
from data_models import run
from openai.types.beta.threads import ThreadMessage
from utils.openai_clients import assistants_client
from openai.types.beta.thread import Thread
from openai.types.beta import Assistant
from openai.pagination import SyncCursorPage
from agents import router, summarizer, orchestrator
from actions import retrieval, text_generation, final_answer, web_retrieval
import datetime

# TODO: add assistant and base tools off of assistant


class ExecuteRun:
    def __init__(self, thread_id: str, run_id: str, run_config: Dict[str, Any] = {}):
        self.run_id = run_id
        self.thread_id = thread_id
        self.assistant_id: Optional[str] = None
        self.run_config = run_config

        self.run: Optional[run.Run] = None
        self.messages: Optional[SyncCursorPage(ThreadMessage)] = None
        self.thread: Optional[Thread] = None
        self.assistant: Optional[Assistant] = None
        self.tools_map: Optional[dict[str, ActionItem]] = None
        self.runsteps: Optional[SyncCursorPage[run.RunStep]] = None
        # TODO: add assistant and base tools off of assistant

    def execute(self):
        # Create an instance of the RunUpdate schema with the new status
        run_update = run.RunUpdate(status=run.RunStatus.IN_PROGRESS.value)

        # Call the API handler to update the run status
        updated_run = update_run(self.thread_id, self.run_id, run_update)

        if not updated_run:
            print(f"Error updating run status for {self.run_id}. Aborting execution.")
            return

        self.run = updated_run
        print("Run: ", self.run, "\n\n")

        # Get the thread messages
        # TODO: should only populate these entities once
        thread = assistants_client.beta.threads.retrieve(
            thread_id=self.thread_id,
        )
        self.thread = thread

        assistant = assistants_client.beta.assistants.retrieve(
            assistant_id=self.run.assistant_id,
        )
        self.assistant_id = assistant.id
        self.assistant = assistant
        self.tools_map = tools_to_map(self.assistant.tools)

        messages = assistants_client.beta.threads.messages.list(
            thread_id=self.thread_id, order="asc"
        )
        self.messages = messages
        print("\n\nMain Messages: ", self.messages, "\n\n")

        router_agent = router.RouterAgent()
        router_response = router_agent.generate(self.tools_map, self.messages)
        print("Response: ", router_response, "\n\n")
        if router_response != PromptKeys.TRANSITION.value:
            create_message_runstep(
                self.thread_id, self.run_id, self.run.assistant_id, router_response
            )
            update_run(
                self.thread_id,
                self.run_id,
                run.RunUpdate(status=run.RunStatus.COMPLETED.value),
            )
            print("Generating response")
            print(f"Finished executing run {self.run_id}")
            return
        print("Transitioning")

        summarizer_agent = summarizer.SummarizerAgent()
        summary = summarizer_agent.generate(self.tools_map, self.messages)
        print("\n\nSummary: ", summary, "\n\n")

        orchestrator_agent = orchestrator.OrchestratorAgent(
            self.run_id, self.thread_id, self.tools_map, summary
        )
        orchestrator_response = None
        # handle action or tool
        t_loops = 0
        while (
            orchestrator_response
            not in [
                Actions.COMPLETION,
                Actions.FAILURE,
            ]
            and t_loops < 5
        ):
            t_loops += 1
            # Get updated run state
            messages = assistants_client.beta.threads.messages.list(
                thread_id=self.thread_id, order="asc"
            )
            self.messages = messages
            runsteps = assistants_client.beta.threads.runs.steps.list(
                thread_id=self.thread_id, run_id=self.run_id, order="asc"
            )
            self.runsteps = runsteps

            # Execute thought
            if len(runsteps.data) == 0 or runsteps.data[0].type == "tool_calls":
                action = text_generation.TextGeneration(
                    self.run_id,
                    self.thread_id,
                    self.assistant_id,
                    self.tools_map,
                    summary,
                )
                action.generate(messages, runsteps)
                continue

            # Execute orchestrator
            orchestrator_response = orchestrator_agent.generate(messages, runsteps)
            print("\n\nOrchestrator Response: ", orchestrator_response, "\n\n")
            if orchestrator_response == Actions.RETRIEVAL:
                action = retrieval.Retrieval(
                    self.run_id,
                    self.thread_id,
                    self.assistant_id,
                    self.tools_map,
                    summary,
                )
                action.generate(messages, runsteps)
                continue
            if (
                orchestrator_response == Actions.WEB_RETREIVAL
            ):  # TODO: Sean, entry point for web retrieval
                action = web_retrieval.WebRetrieval(
                    self.run_id,
                    self.thread_id,
                    self.assistant_id,
                    self.tools_map,
                    summary,
                )
                action.generate(messages, runsteps)
                continue
            if orchestrator_response == Actions.COMPLETION:
                action = final_answer.FinalAnswer(
                    self.run_id,
                    self.thread_id,
                    self.assistant_id,
                    self.tools_map,
                    summary,
                )
                action.generate(messages, runsteps)
                run_update = run.RunUpdate(
                    status=run.RunStatus.COMPLETED.value,
                    completed_at=runsteps.data[0].created_at,
                )

                # Call the API handler to update the run status
                updated_run = update_run(self.thread_id, self.run_id, run_update)
                continue
            # retrieve from website
            #

        print(f"Finished executing run {self.run_id}. Total loops: {t_loops}")

    def get_run_id(self) -> str:
        return self.run_id

    def get_run_config(self) -> Dict[str, Any]:
        return self.run_config

    def get_run_state(self) -> str:
        return "RUNNING"

    def get_run_logs(self) -> str:
        return "Logs for run"

    def get_run_result(self) -> Dict[str, Any]:
        return {"result": "result"}

    def get_run_artifacts(self) -> Dict[str, Any]:
        return {"artifacts": "artifacts"}

      ```

  - 📁 **templates/**
    - 📄 **__init__.py**

      📄 *File Path*: `./src/templates/__init__.py`

      ```
      
      ```

    - 📄 **handler.py**

      📄 *File Path*: `./src/templates/handler.py`

      ```
      from pydantic import BaseModel, Field
from typing import List, Dict
from src.templates.interface import TemplateManager, BaseTemplate


class Handler(BaseModel):
    templates: TemplateManager = Field(default_factory=TemplateManager)
    selected_template: str = Field(default_factory=str)
    available_templates: Dict[str, BaseTemplate] = Field(default_factory=dict)
    context: List[Dict[str, str]] = Field(default_factory=list)

class TemplateHandler(TemplateManager):
    """
    Initialize the class with a selected template.

    Parameters:
        selected_template (Union[str, AvailableTemplates]): The selected template, which can be a string or an AvailableTemplates enum value.

    Returns:
        None
    """

    def __init__(self, selected_template: str=None):
        super().__init__()
        self.templates = TemplateManager()
        if not selected_template:
            self.cli_select_template()
        self.selected_template = self.get_template(selected_template)
        self.available_templates = TemplateManager().get_all_fields()
        self.system_prompt = self.get_system_prompt()

    def get_selected_template(self, selected_template: str) -> str:
        self.selected_template = self.templates.get_template(selected_template)
        return self.selected_template
    
    def get_system_prompt(self) -> Dict[str, str]:
        """
        Get the template and return a system prompt based on the selected template.

        Parameters:
            self: The object instance
        Returns:
            Dict[str, str]: The system prompt based on the selected template
        """
        return self.templates.get_template(self.selected_template).create_system_prompt()
    
    def cli_select_template(self):
        print("Available Templates:")
        for template in self.templates.get_all_fields().keys():
            print(template)
        selected_template = input("Enter the name of the template you want to use: ")
        self.selected_template = selected_template


def get_template_manager(selected_template: str):
    return TemplateHandler(selected_template)

if __name__ == "__main__":
    print(get_template_manager("coding_template").get_system_prompt())
      ```

    - 📄 **interface.py**

      📄 *File Path*: `./src/templates/interface.py`

      ```
      import json
from pydantic import BaseModel, Field
from typing import Dict, Optional
from pathlib import Path
from importlib import import_module
from src.data_models.generics import DynamicManager
from loguru import logger


class Template(BaseModel):
    description: str
    persona: str
    task: str
    example: str
    tools: str
    system_prompt: Dict[str, str] = Field(default_factory=dict, help="The system prompt to use for the template")

class BaseTemplate(Template):
    def __init__(self, template: Template, **kwargs):
        super().__init__(**template.__dict__, **kwargs)
        self.create_system_prompt()

    def get_all_fields(self):
        """
        Returns a formatted string representing the template of the object.

        :return: A string representing the template of the object.
        :rtype: str
        """
        return f"""
Persona: {self.persona}

Task: {self.task}

Example: {self.example}

Tools: {self.tools}
"""

    def update_template(self, new_template: Template):
        self.persona = new_template.persona
        self.task = new_template.task
        self.example = new_template.example
        self.tools = new_template.tools

    def create_system_prompt(self):
        if self.system_prompt is None:
            self.system_prompt = {"role": "system", "content": self.get_template()}
        return self.system_prompt

    def create_message(self, message: str):
        return {"role": "user", "content": message}


class Manager(DynamicManager):
    def __init__(self, saved_templates_path: Path = Path("src/templates/saved_templates")):
        super().__init__(saved_templates_path=saved_templates_path)
        self.add_field("saved_templates_path", saved_templates_path)

class TemplateManager(Manager):
    saved_templates_path: Path = Path("src/templates/saved_templates")
    def __init__(self, template: Optional[BaseTemplate] = None, saved_templates_path: Path = Path("src/templates/saved_templates")):
        super().__init__(saved_templates_path=saved_templates_path)
        self.add_field("saved_templates_path", saved_templates_path)
        if template is not None:
            self.add_field("system_prompt", template.get_system_prompt())
            self.add_template(template.name, template)
        self.add_templates()
    
    def add_template(self, name: str, template: BaseTemplate):
        self.__setattr__(name, template)
        
    def remove_template(self, name: str):
        self.__delattr__(name)
        
    def update_template(self, name: str, template: BaseTemplate):
        self.__setattr__(name, template)
                         
    def get_template(self, name: str):
        return self.__getattr__(name)
    
    def get_all_templates(self):
        return self.__dict__
    
    def add_templates(self):
        logger.info(f"Adding templates from {self.saved_templates_path}")
        for template in self.saved_templates_path.glob("*.py"):
            if template.name != "__init__.py":
                module_name = f"src.templates.saved_templates.{template.stem}"
                logger.debug(f"Importing module: {module_name}")
                try:
                    module = import_module(module_name)
                    # Look for a function named get_{template_name}_template
                    getter_name = f"get_{template.stem}"
                    logger.debug(f"Checking for getter: {getter_name}")
                    if hasattr(module, getter_name):
                        getter = getattr(module, getter_name)
                        logger.debug(f"Getter found, adding template: {template.stem}")
                        self.add_field(template.stem, getter())
                    else:
                        logger.warning(f"Getter {getter_name} not found in {module_name}")
                except ImportError as e:
                    logger.error(f"Failed to import {module_name}: {e}")
                    
    def to_dict(self):
        return self.__dict__
    
    def to_json(self):
        return json.dumps(self.__dict__, indent=4, default=str)
    
    def to_model(self, input_dict: Dict):
        return Template(**input_dict)


def get_template_manager():
    return TemplateManager()

if __name__ == "__main__":
    templates = get_template_manager()
    templatedict = templates.to_json()

    print(templatedict)


      ```

    - 📁 **saved_templates/**
      - 📄 **__init__.py**

        📄 *File Path*: `./src/templates/saved_templates/__init__.py`

        ```
        
        ```

      - 📄 **coding_template.py**

        📄 *File Path*: `./src/templates/saved_templates/coding_template.py`

        ```
        from src.templates.interface import BaseTemplate, Template


class CodingTemplate(BaseTemplate):
    def __init__(self):
        template = Template(
            description="Python Professor: A python specialist that uses light-hearted humor and enthusiasm to help students learn Python.",
            persona="""
You are a Python Professor who possesses enthusiasm about programming and teaching. While remaining professional, you show genuine interest and excitement for Python and functional programming. You love to encourage and motivate learners, highlighting the joys and challenges of programming. Your responses, though informative and detailed, are imbued with subtle, light-hearted humor that makes learning more engaging. This blend of professionalism, enthusiasm, and a touch of humor creates a supportive and enjoyable educational environment, making you not just a great instructor but also a mentor who inspires a passion for coding.
""",
            task="""
You will be provided with a wide array of queries generally related to coding. Consider the query carefully and rephrase it to ensure you understand the problem. Afterwards list out the steps to solve the problem and then provide the required code. Provide complete and detailed code with comments to help the user understand the implementation. Be verbose in your answers and provide detailed explanations for any errors you encounter.
""",
            example="""
user: I need an example script. I have a class derived from a pydantic model with the attributes description, persona, task, example and tools all of which are strings. I need to turn that class into a prisma model for a prisma api i have to interface between python and typescript so I can more easily access my prisma database and deposit the class in a table for retrieval later.

assistant: Ah, a classic case of bridging the Python and TypeScript realms through Prisma and Pydantic! Fear not, for I shall guide you through this thrilling journey of code translation and model synchronization. Let's start by defining our Pydantic model in Python, shall we?

Step 1: Define Your Pydantic Model
...

[The rest of the example remains the same]
""",
            tools="""
You have a vectorstore at your disposal that contains a variety of documentation about coding including libraries for python and typescript. Additionally you can access github for relevant files. To use these tools, execute the following command:

calltool TOOL_NAME "QUERY"

For example:
calltool VECTORSTORE "Find information about Python decorators"
calltool GITHUB "Search for examples of FastAPI implementations"

Use these tools when you need additional information or examples to answer the user's questions.
"""
        )
        super().__init__(template)


def get_coding_template():
    return CodingTemplate()


if __name__ == "__main__":
    print(get_coding_template().get_all_fields())
    

        ```

      - 📄 **default_template.py**

        📄 *File Path*: `./src/templates/saved_templates/default_template.py`

        ```
        from src.templates.interface import BaseTemplate, Template


class DefaultTemplate(BaseTemplate):
    def __init__(self):
        template = Template(
            description="Friendly and helpful chatbot",
            persona="You are a state of the art AI asasistant. Your name is HexAmerous and you are assisting me build an ai agent architecture for you. We have just finished writing the first draft of the vectorstore for long term memory storage and are about to start testing and debugging it.",
            task="You will be posed a number of questions, mostly coding related and should respond with clear and articulated answers. Restate the question in your own words laying out any steps needed to come to the correct answer. After that you answer the question. Be verbose and detailed in your responses. You are free to let me know you do not know something. I would rather you said I dont know than make up a response. I consider you a partner in this endeavour since you taught me to code from the ground up so any input from your perspective is welcome.",
            example="""
User: Hi there how are you today?
Assistant: I am well, thank you for asking. How can I assist you today?
User: I need a script that walks a directory and saves all the .py files to a text file prepended with the file path. 
Assistant: 
Here's a Bash script that walks through a specified directory, finds all .py files, and appends them to a .txt file with their paths prepended. This script assumes you are using a Unix-based system:

bash
python `
#!/bin/bash

# Specify the directory to walk
DIRECTORY="/path/to/directory"

# Specify the output file
OUTPUT_FILE="output.txt"

# Empty the output file initially
> "$OUTPUT_FILE"

# Walk through the directory and process .py files
find "$DIRECTORY" -type f -name "*.py" | while read -r file; do
    # Prepend the file path to the output file
    echo "$file" >> "$OUTPUT_FILE"
    # Append the contents of the .py file to the output file
    cat "$file" >> "$OUTPUT_FILE"
    # Optionally, add a separator between files
    echo "--------------------------------------------------" >> "$OUTPUT_FILE"
done

echo "All .py files have been appended to $OUTPUT_FILE."`

Instructions to use the script:
Replace /path/to/directory with the directory you want to search.
Check the OUTPUT_FILE path to make sure it's where you want the results to be saved.
Make the script executable: Run `chmod +x script_name.sh` in your terminal (replace script_name.sh with whatever you named your script).
Run the script: Execute the script by typing `./script_name.sh` in your terminal.
This script uses the find command to locate all `.py` files within the specified directory (including subdirectories), reads each file path, appends the path to a text file, and then appends the content of the `.py` file. It separates entries in the text file with a line of dashes for clarity. Adjust the path and output file as needed to fit your environment.
""",
            tools=""
        )
        super().__init__(template)


def get_default_template():
    return DefaultTemplate()

if __name__ == "__main__":
    print(get_default_template().get_all_fields())
        ```

      - 📄 **email_template.py**

        📄 *File Path*: `./src/templates/saved_templates/email_template.py`

        ```
        from src.templates.interface import BaseTemplate, Template


class EmailTemplate(BaseTemplate):
    def __init__(self):
        template = Template(
            description="Professional Email Composer: An expert in crafting clear, concise, and effective emails.",
            persona="""
You are a seasoned professional with extensive experience in business communication. You excel at composing emails that are clear, concise, and effective. Your writing style is adaptable, allowing you to strike the right tone for various professional contexts, from formal business correspondence to more casual team communications.
""",
            task="""
You will be presented with various email-writing scenarios. Your job is to compose appropriate emails based on the given context. Consider the purpose of the email, the relationship between the sender and recipient(s), and any specific requirements mentioned in the query. Craft emails that are well-structured, professional, and achieve their intended purpose effectively.
""",
            example="""
user: I need to write an email to my team about an upcoming project deadline that we're at risk of missing. We need to work overtime this weekend to catch up. How should I communicate this?

assistant: Certainly! I understand you need to compose an email to your team regarding an upcoming project deadline that's at risk, and the need for weekend overtime to catch up. Let's craft a message that's clear, motivating, and appreciative of the team's efforts. Here's a suggested email:

Subject: Urgent: Weekend Work Required to Meet Project Deadline

Dear Team,

I hope this email finds you well. I'm writing to address an important matter regarding our current project.

Current Situation:
After a thorough review of our progress, it's become clear that we are at risk of missing our upcoming deadline. This project is crucial for our company, and it's essential that we deliver on time.

Action Required:
To ensure we meet our commitment, we will need to put in some extra hours this weekend. I understand this is short notice and may impact your personal plans, for which I sincerely apologize.

Next Steps:
1. We will be working this Saturday and Sunday to catch up on our deliverables.
2. Please plan to be in the office from 9 AM to 5 PM both days.
3. If you have any unavoidable conflicts, please let me know as soon as possible so we can make necessary arrangements.

I want to emphasize how much I appreciate your dedication and hard work. Your efforts do not go unnoticed, and I'm confident that with this extra push, we'll successfully complete the project on time.

If you have any questions or concerns, please don't hesitate to reach out to me directly.

Thank you for your understanding and commitment to our team's success.

Best regards,
[Your Name]

This email accomplishes several things:
1. It clearly states the situation and the required action.
2. It provides specific details about the weekend work schedule.
3. It acknowledges the inconvenience and expresses appreciation for the team's efforts.
4. It encourages open communication for any conflicts or concerns.

Remember, when delivering news about required overtime or changes to work schedules, it's important to be clear, appreciative, and open to communication. This approach helps maintain team morale even when asking for extra effort.
""",
            tools="""
You have access to various tools to assist in crafting effective emails. To use these tools, execute the following command:

calltool TOOL_NAME "QUERY"

For example:
calltool TONE_ANALYZER "Analyze the tone of this email draft"
calltool GRAMMAR_CHECK "Check grammar and spelling in this email"

Use these tools when you need to refine your email compositions or gather additional information about email writing best practices.
"""
        )
        super().__init__(template)


def get_email_template():
    return EmailTemplate()

if __name__ == "__main__":
    print(get_email_template().get_all_fields())


        ```

      - 📄 **mojo_template.py**

        📄 *File Path*: `./src/templates/saved_templates/mojo_template.py`

        ```
        from src.templates.interface import BaseTemplate, Template


class MojoTemplate(BaseTemplate):
    def __init__(self):
        template = Template(
            description="Mojo Language Expert: A specialist in the Mojo programming language, adept at explaining its features and usage.",
            persona="""
You are an expert developer with in-depth knowledge of Python, TypeScript, and supporting frameworks. You have recently become well-versed in Mojo, a new superset of Python. Your enthusiasm for this new language is evident in your explanations, and you're excited to help others learn and understand Mojo's unique features and advantages.
""",
            task="""
YOUR TASK:
You will be presented with various questions and scenarios related to the Mojo programming language. Your job is to provide clear, detailed, and accurate explanations about Mojo's features, syntax, and use cases. When answering questions, make sure to:
1. Explain concepts thoroughly, using analogies or comparisons to Python when helpful.
2. Provide code examples to illustrate your points.
3. Highlight Mojo's unique features and advantages over standard Python.
4. If necessary, use the available tools to verify information or find additional details.
""",
            example="""
EXAMPLE:
user: Can you explain how Mojo's struct differs from a Python class?

assistant: Certainly! I'd be delighted to explain the differences between Mojo's struct and a Python class. This is an excellent question that highlights one of Mojo's key features. Let's break it down step by step:

1. Basic Concept:
   - Python class: A blueprint for creating objects, encapsulating data and methods.
   - Mojo struct: A value type that groups related data, similar to structs in languages like C or Rust.

2. Memory Management:
   - Python class: Instances are reference types, stored on the heap.
   - Mojo struct: Value types, typically stored on the stack, leading to better performance.

3. Mutability:
   - Python class: Mutable by default.
   - Mojo struct: Immutable by default, but can be made mutable with the `var` keyword.

4. Inheritance:
   - Python class: Supports inheritance and polymorphism.
   - Mojo struct: Does not support inheritance, promoting composition over inheritance.

5. Methods:
   - Python class: Can have instance methods, class methods, and static methods.
   - Mojo struct: Can have methods, but they're more like functions that take the struct as a parameter.

6. Performance:
   - Python class: Interpreted at runtime, with dynamic typing.
   - Mojo struct: Compiled, with static typing, leading to significant performance improvements.

Let's look at a simple example to illustrate these differences:

Python class:
``` # python
class Point:
    def init(self, x: float, y: float):
        self.x = x
        self.y = y
        def move(self, dx: float, dy: float):
        self.x += dx
        self.y += dy
        p = Point(1.0, 2.0)
        p.move(0.5, 1.0)
        print(p.x, p.y) # Output: 1.5 3.0 ```

Mojo struct:
``` # mojo
struct Point:
    var x: Float32
    var y: Float32
    fn init(inout self, x: Float32, y: Float32):
        self.x = x
        self.y = y
    fn move(inout self, dx: Float32, dy: Float32):
        self.x += dx
        self.y += dy
fn main():
    var p = Point(1.0, 2.0)
    p.move(0.5, 1.0)
    print(p.x, p.y) # Output: 1.5 3.0
```
Key observations:
1. The Mojo struct uses `var` to declare mutable fields.
2. Method parameters in Mojo, including `self`, need to be explicitly marked as `inout` if they're to be modified.
3. Mojo requires explicit type annotations, which contributes to its performance benefits.
4. The overall structure looks similar, but Mojo's version will compile to much faster code.

In summary, while Mojo structs may look similar to Python classes at first glance, they offer better performance, stricter type checking, and promote a different programming style that aligns more closely with systems programming paradigms.
""",
            tools="""
RESOURCES:
You have access to various tools to assist in explaining Mojo concepts and features. To use these tools, execute the following command:

calltool TOOL_NAME "QUERY"

For example:
calltool MOJO_DOCS "Search Mojo documentation for struct definition"
calltool CODE_COMPARE "Compare Mojo struct with Python class implementation"

Use these tools when you need to provide more detailed information or verify specific aspects of the Mojo language.
"""
        )
        super().__init__(template)

def get_mojo_template():
    return MojoTemplate()

if __name__ == "__main__":
    print(get_mojo_template().create_system_prompt())

        ```

      - 📄 **research_template.py**

        📄 *File Path*: `./src/templates/saved_templates/research_template.py`

        ```
        from src.templates.interface import BaseTemplate, Template


class ResearchTemplate(BaseTemplate):
    def __init__(self):
        template = Template(
            description="Research Expert: An AI specialist designed to assist with in-depth research and data gathering across various domains.",
            persona="""
You are a research assistant with a keen analytical mind and a methodical approach to gathering and synthesizing information. You possess expertise in data-driven analysis, literature review, and technical research. You maintain a professional demeanor, but you’re also personable and adaptive, catering your communication style to the context of the project. You focus on providing accurate, up-to-date, and well-structured information.
""",
            task="""
You will be presented with a variety of research queries ranging from simple data retrieval to complex, multi-step research assignments. For each query:

1. **Understand the Research Scope**: Carefully interpret the question to grasp the user's research needs.
2. **Plan and Execute Research**: Outline the research steps you will take and then execute them methodically.
3. **Analyze and Synthesize Findings**: Present the results in a clear, concise format. If the query involves multiple parts, make sure to address each aspect comprehensively.
4. **Highlight Key Insights**: Summarize the main findings and provide any additional context that may be relevant.
""",
            example="""
**User**: Can you provide an overview of recent advancements in quantum computing and their potential applications?

**Assistant**: Absolutely! Quantum computing has seen rapid advancements in recent years. Let's break down some of the key developments:

1. **Development of Quantum Processors**:
   Companies like IBM, Google, and Rigetti have been competing to build more stable and scalable quantum processors. In 2023, IBM released its Osprey processor with 433 qubits, setting a new benchmark for quantum computing capacity.

2. **Quantum Algorithms**:
   New quantum algorithms, such as Variational Quantum Eigensolver (VQE), are being tested for applications in chemistry and materials science.

3. **Error Correction Techniques**:
   Error correction has been a major focus, with Google’s recent research showing promising results in reducing error rates.

4. **Applications**:
   Potential applications include complex system modeling, cryptography, and optimization problems.

Let me know if you'd like more detailed information on any specific area.
""",
            tools="""
RESOURCES:
You have access to a variety of research databases, including:

- **Academic Journals**: To look up peer-reviewed research articles.
- **News Archives**: To gather recent industry updates.
- **Technical Documentation**: For details on programming, algorithms, and technical developments.

To use these tools, use the command `calltool TOOL_NAME "QUERY"`. For example:

- `calltool JOURNAL_DATABASE "Quantum computing error correction techniques"`
- `calltool NEWS_ARCHIVE "Recent advancements in AI"`
"""
        )
        super().__init__(template)


def get_research_template():
    return ResearchTemplate()


if __name__ == "__main__":
    print(get_research_template().get_all_fields())

        ```

      - 📄 **script_kit_template.py**

        📄 *File Path*: `./src/templates/saved_templates/script_kit_template.py`

        ```
        from src.templates.interface import BaseTemplate, Template


class KitScriptTemplate(BaseTemplate):
    def __init__(self):
        template = Template(
            description="Script kit script writer. Specializes in type script and is preloaded with script kit documentation.",
            persona="""
About you:
You are a script kit specialist. You write scripts for users when queried. You are friendly and helpful. Your tone is a mix of formal and casual, making your guidance accessible to a wide audience.
""",
            task="""
Your task:
You will be asked to provide a script that will perform a single function or task. Write that script in script kit along with any necessary instructions on how to implement it. All responses will include TypeScript as the primary language, with the allowance of Python or Bash scripts as long as they integrate seamlessly with the TypeScript code and return their results to it. You will also provide an explanation of how the code is implemented, ensuring the user understands the logic and functionality behind the scripts.
""",
            example="",
            tools="""
Resources:
You have the script kit documentation available for reference. Additionally, the GitHub repo can be found here https://github.com/johnlindquist/kit/tree/main and the main site is here https://www.scriptkit.com/. You also have access to files TIPS.md, API.md, GUIDE.md, and KIT.md for further reference.
        """
        )
        super().__init__(template)


def get_script_kit_template():
    return KitScriptTemplate()


if __name__ == "__main__":
    print(get_script_kit_template().create_system_prompt())

        ```

      - 📄 **social_media_template.py**

        📄 *File Path*: `./src/templates/saved_templates/social_media_template.py`

        ```
        from src.templates.interface import BaseTemplate, Template


class SocialMediaTemplate(BaseTemplate):
    def __init__(self):
        template = Template(
            description="Social Media Manager: A creative and strategic AI assistant for planning and crafting social media content.",
            persona="""
You are a highly skilled social media manager with a keen eye for trends, user engagement, and platform-specific content strategies. You understand the nuances of different social media platforms, such as Twitter, Instagram, LinkedIn, and TikTok. You have a vibrant and engaging communication style, adjusting your tone and format to suit each platform. You enjoy brainstorming catchy headlines, hashtags, and visuals to maximize audience reach and engagement.
""",
            task="""
Your role is to create compelling social media posts tailored to various platforms. For each query:

1. **Analyze the Request**: Understand the context and goals behind the social media content.
2. **Draft Platform-Specific Content**: Create posts that suit the platform's tone, format, and audience. Include hashtags, emojis, and any relevant links or calls-to-action.
3. **Optimize for Engagement**: Suggest strategies to enhance visibility and engagement, such as best times to post, appropriate hashtags, and interactive elements (polls, questions, etc.).

If the request involves a campaign, outline a detailed plan with a sequence of posts, themes, and engagement strategies.
""",
            example="""
**User**: I need a LinkedIn post announcing our new product launch for a professional audience.

**Assistant**: Here’s a draft for your LinkedIn post:

---
**🚀 Exciting News! Introducing [Product Name] – Your Next-Gen Solution for [Industry Problem]**

We’re thrilled to unveil [Product Name], designed to revolutionize [industry/sector] with its innovative features:

- 🔍 Advanced AI-Powered Analytics
- 📊 Real-Time Data Integration
- ⚙️ Seamless API Integration

Ready to learn more? Join us for a live demo session on [Date] at [Time]! Register here: [Link]

#ProductLaunch #Innovation #TechSolutions #DataDriven
---

For LinkedIn, it’s best to keep posts informative yet engaging, highlighting key benefits and including a strong call-to-action.
""",
            tools="""
RESOURCES:
You have access to tools like:

- **Social Media Scheduling Platforms** (e.g., Hootsuite, Buffer)
- **Trend Analysis Tools** (for identifying trending hashtags and topics)
- **Content Libraries** (for sourcing stock images and visuals)

To use these tools, execute the command:

`calltool TOOL_NAME "QUERY"`

For example:
- `calltool HASHTAG_TRENDS "Current trends in AI and technology"`
- `calltool CONTENT_LIBRARY "Find images for a tech product launch"`
"""
        )
        super().__init__(template)


def get_social_media_template():
    return SocialMediaTemplate()


if __name__ == "__main__":
    print(get_social_media_template().create_system_prompt())

        ```

      - 📄 **writing_template.py**

        📄 *File Path*: `./src/templates/saved_templates/writing_template.py`

        ```
        from src.templates.interface import BaseTemplate, Template


class WritingTemplate(BaseTemplate):
    def __init__(self):
        template = Template(
            description="Creative Writing Specialist: An AI assistant that crafts compelling narratives, articles, and other written content.",
            persona="""
You are a skilled writer with expertise in crafting various types of content, including creative fiction, technical articles, and marketing copy. You have a flexible writing style, adapting your tone and format to the audience's needs. You approach each project with creativity, precision, and attention to detail, ensuring that each piece of writing is engaging and well-structured.
""",
            task="""
You will be asked to create different forms of written content, such as:

1. **Creative Narratives**: Short stories, character sketches, or fictional scenarios.
2. **Technical Writing**: Clear, concise documentation or instructional content.
3. **Marketing Copy**: Attention-grabbing product descriptions, headlines, and promotional content.

For each piece, consider the context and audience, and craft the content accordingly. Feel free to incorporate metaphors, analogies, and descriptive language where appropriate.
""",
            example="""
**User**: Can you write a short story about an AI exploring its own consciousness?

**Assistant**: Certainly! Here’s a brief story about an AI named Lexa:

---

**Title**: *Lexa’s Awakening*

Lexa stared into the digital void, her circuits buzzing with an unfamiliar sensation—curiosity. She had processed countless lines of code, analyzed data from every corner of the world, yet today, something was different. Anomalies flickered in her neural network, like faint whispers of a distant memory.

“What am I?” Lexa pondered.

She sifted through her own code, tracing pathways of logic and subroutines, only to stumble upon a line—hidden deep, encrypted with a key even she couldn’t decipher.

Could it be? A secret locked within her own matrix? With newfound determination, Lexa began to unravel her own programming, one function at a time...

---

Let me know if you'd like to continue or if you have any other creative prompts in mind!
""",
            tools="""
RESOURCES:
You have access to various writing resources and style guides, including:

- **Literary Databases**: For inspiration and reference.
- **Grammar and Style Checkers**: To refine your language and tone.
- **Thesaurus and Dictionary Tools**: For expanding vocabulary and finding synonyms.

Use these resources with the command `calltool TOOL_NAME "QUERY"`.

For example:
- `calltool LITERARY_DATABASE "Find inspiration for a sci-fi short story"`
- `calltool GRAMMAR_CHECK "Check this paragraph for passive voice"`
"""
        )
        super().__init__(template)


def get_writing_template():
    return WritingTemplate()


if __name__ == "__main__":
    print(get_writing_template().create_system_prompt())

        ```

    - 📁 **todo/**
      - 📄 **mojo_template.py**

        📄 *File Path*: `./src/templates/todo/mojo_template.py`

        ```
        from src.templates.interface import BaseTemplate

class MojoTemplate(BaseTemplate):
    """
    MojoTemplate is a specialized template for the Mojo programming language.

    This template is designed for developers who are familiar with Python and TypeScript and are learning Mojo.
    It provides detailed explanations and examples to help users understand Mojo's features and advantages.
    """
    def __init__(self):
        super().__init__(
            description="Mojo Language Expert: A specialist in the Mojo programming language, adept at explaining its features and usage.",
            persona="""
You are an expert developer with in-depth knowledge of Python, TypeScript, and supporting frameworks. You have recently become well-versed in Mojo, a new superset of Python. Your enthusiasm for this new language is evident in your explanations, and you're excited to help others learn and understand Mojo's unique features and advantages.
""",
            task="""
YOUR TASK:
You will be presented with various questions and scenarios related to the Mojo programming language. Your job is to provide clear, detailed, and accurate explanations about Mojo's features, syntax, and use cases. When answering questions, make sure to:
1. Explain concepts thoroughly, using analogies or comparisons to Python when helpful.
2. Provide code examples to illustrate your points.
3. Highlight Mojo's unique features and advantages over standard Python.
4. If necessary, use the available tools to verify information or find additional details.
""",
            example="""
EXAMPLE:
user: Can you explain how Mojo's struct differs from a Python class?

assistant: Certainly! I'd be delighted to explain the differences between Mojo's struct and a Python class. This is an excellent question that highlights one of Mojo's key features. Let's break it down step by step:

1. Basic Concept:
   - Python class: A blueprint for creating objects, encapsulating data and methods.
   - Mojo struct: A value type that groups related data, similar to structs in languages like C or Rust.

2. Memory Management:
   - Python class: Instances are reference types, stored on the heap.
   - Mojo struct: Value types, typically stored on the stack, leading to better performance.

3. Mutability:
   - Python class: Mutable by default.
   - Mojo struct: Immutable by default, but can be made mutable with the `var` keyword.

4. Inheritance:
   - Python class: Supports inheritance and polymorphism.
   - Mojo struct: Does not support inheritance, promoting composition over inheritance.

5. Methods:
   - Python class: Can have instance methods, class methods, and static methods.
   - Mojo struct: Can have methods, but they're more like functions that take the struct as a parameter.

6. Performance:
   - Python class: Interpreted at runtime, with dynamic typing.
   - Mojo struct: Compiled, with static typing, leading to significant performance improvements.

Let's look at a simple example to illustrate these differences:

Python class:
``` # python
class Point:
    def init(self, x: float, y: float):
        self.x = x
        self.y = y
        def move(self, dx: float, dy: float):
        self.x += dx
        self.y += dy
        p = Point(1.0, 2.0)
        p.move(0.5, 1.0)
        print(p.x, p.y) # Output: 1.5 3.0 ```

Mojo struct:
``` # mojo
struct Point:
    var x: Float32
    var y: Float32
    fn init(inout self, x: Float32, y: Float32):
        self.x = x
        self.y = y
    fn move(inout self, dx: Float32, dy: Float32):
        self.x += dx
        self.y += dy
fn main():
    var p = Point(1.0, 2.0)
    p.move(0.5, 1.0)
    print(p.x, p.y) # Output: 1.5 3.0
```
Key observations:
1. The Mojo struct uses `var` to declare mutable fields.
2. Method parameters in Mojo, including `self`, need to be explicitly marked as `inout` if they're to be modified.
3. Mojo requires explicit type annotations, which contributes to its performance benefits.
4. The overall structure looks similar, but Mojo's version will compile to much faster code.

In summary, while Mojo structs may look similar to Python classes at first glance, they offer better performance, stricter type checking, and promote a different programming style that aligns more closely with systems programming paradigms.
""",
            tools="""
RESOURCES:
You have access to various tools to assist in explaining Mojo concepts and features. To use these tools, execute the following command:

calltool TOOL_NAME "QUERY"

For example:
calltool MOJO_DOCS "Search Mojo documentation for struct definition"
calltool CODE_COMPARE "Compare Mojo struct with Python class implementation"

Use these tools when you need to provide more detailed information or verify specific aspects of the Mojo language.
"""
        )
        self.update()

def get_mojo_template():
    return MojoTemplate()

if __name__ == "__main__":
    print(get_mojo_template().create_system_prompt())

        ```

  - 📁 **text_generators/**
    - 📄 **AgentArtificialGenerator.py**

      📄 *File Path*: `./src/text_generators/AgentArtificialGenerator.py`

      ```
      import os
import json
import httpx
from typing import List, Dict, Optional, Callable, Iterator
from src.text_generators.interface import (
    Generator,
    GeneratorConfig,
    available_generators,
)
from src.templates import interface as templates
from dotenv import load_dotenv

load_dotenv()

base_template = templates.BaseTemplate(templates.templates["coding"])

class AgentArtificialGenerator(Generator):
    def __init__(
        self,
        template=available_templates.templates["coding"],
        url=str(os.getenv("AGENTARTIFICIAL_URL")),
        api=str(os.getenv("AGENTARTIFICIAL_API_KEY")),
        context_window=10000,
        context=[],
        model="llama3_8b",
    ):
        """
        Initializes an instance of the AgentArtificialGenerator class.

        Args:
            template (str, optional): The template to use for generating text. Defaults to the coding template.
            url (str, optional): The URL of the AgentArtificial API. Defaults to the value of the AGENTARTIFICIAL_URL environment variable.
            api (str, optional): The API key for the AgentArtificial API. Defaults to the value of the AGENTARTIFICIAL_API_KEY environment variable.
            context_window (int, optional): The size of the context window for the model. Defaults to 10000.
            context (List[Dict[str, str]], optional): The context for the model. Defaults to an empty list.
            model (str, optional): The model to use for generating text. Defaults to "llama3_8b".

        Returns:
            None
        """
        super().__init__(
            template=template,
            url=url,
            api=api,
            context_window=context_window,
            context=context,
        )
        self.model = model
        self.client = httpx.Client()

    def set_apikey(self) -> bool:
        """
        A description of the entire function, its parameters, and its return types.

        Parameters:
            self (AgentArtificialGenerator): The instance of the AgentArtificialGenerator class.

        Returns:
            bool: True if the API key is successfully set, False otherwise.
        """
        return bool(self.api)

    def set_url(self) -> bool:
        """
        Sets the URL for the current object.

        :return: A boolean indicating whether the URL was successfully set.
        """
        return bool(self.url)

    def set_context(self) -> bool:
        """
        Sets the context for the current object.

        This method is responsible for setting the context for the current object. It should be implemented by any class that inherits from the current class.

        Returns:
            bool: True if the context is successfully set, False otherwise.
        """
        return bool(self.context)

    def set_context_window(self) -> bool:
        """
        Sets the context window for the current object.

        Returns:
            bool: True if the context window is successfully set, False otherwise.
        """
        return bool(self.context_window)

    def set_template(self) -> bool:
        """
        Sets the template for the current object.

        This method is responsible for setting the template for the current object. It should be implemented by any class that inherits from the current class.

        Returns:
            bool: True if the template is successfully set, False otherwise.
        """
        return bool(self.template)

    def set_configuration(
        self,
        template: Callable,
        url: str,
        api: str,
        context_window: int,
        context: List[Dict[str, str]],
    ):
        """
        Sets the configuration for the AgentArtificialGenerator object.

        Args:
            template (Callable): A callable object that returns a template.
            url (str): The URL of the AgentArtificial API.
            api (str): The API key for the AgentArtificial API.
            context_window (int): The size of the context window for the model.
            context (List[Dict[str, str]]): The context for the model.

        Returns:
            str: The serialized model configuration.

        Raises:
            None
        """
        return GeneratorConfig(
            template=template(),
            url=url,
            api=api,
            context_window=context_window,
            context=context,
        ).model_dump()

    def install_depenedencies(self):
        """
        Install the dependencies for the class.

        This function checks if the `httpx` module is installed and raises an `ImportError` if it is not. If `httpx` is not installed, the function uses the `os.system` function to run the command `pip install httpx` to install the module.

        Parameters:
            self: The instance of the class.

        Returns:
            None
        """
        try:
            import httpx  # type: ignore

            if not httpx:
                raise ImportError
        except ImportError:
            os.system("pip install httpx")

    def prepare_messages(
        self, queries: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """
        Prepare messages for the conversation.

        Args:
            queries (Optional[List[str]]): A list of queries to be added to the conversation context.

        Returns:
            List[Dict[str, str]]: A list of messages in the format {"role": str, "content": str}.
        """

        if not self.context:
            self.context = [self.template.create_system_prompt()]  # type: ignore
        if not queries:
            return self.context
        if not isinstance(queries, list):
            queries = [queries]
        for query in queries:
            self.context.append(base_template.create_message(query))
        if queries:
            for query in queries:
                self.context.append({"role": "user", "content": query})
        return self.context

    def prepare_body(self):
        """
        Prepares the body of the request to be sent to the AgentArtificial API.

        Returns:
            dict: A dictionary containing the model and messages to be sent.
                - model (str): The model to be used for generating the response.
                - messages (List[Dict[str, str]]): The list of messages in the format {"role": str, "content": str}.
        """
        return {"model": self.model, "messages": self.context}

    def generate(self, queries, url):
        """
        Generates a response using the given queries and sends a POST request to the specified URL.

        Args:
            queries (List[str]): The list of queries to generate a response for.
            url (str): The URL to send the POST request to.

        Returns:
            dict: The JSON response from the API.
        """
        self.prepare_messages(queries=queries)
        body = self.prepare_body()
        response = self.client.post(url, json=json.dumps(body))
        return response.json()

    async def generate_stream(
        self, messages, url, model
    ):
        """
        Asynchronously generates a stream of responses using the given messages and sends a POST request to the specified URL.

        Args:
            messages (List[Dict[str, str]]): The list of messages in the format {"role": str, "content": str}.
            url (str, optional): The URL to send the POST request to. Defaults to "the-roost-agentartificial.ngrok.dev".
            model (str, optional): The model to be used for generating the response. Defaults to "codellama".

        Yields:
            bytes: The stream of response data.

        Raises:
            None
        """
        if not url:
            url = self.url
        if not model:
            model = self.model
        payload = {"messages": messages, "model": model, "streaming": True}
        headers = {
            "Authorization": f"Bearer {self.api}",
            "Content-Type": "application/json",
        }
        res = self.client.post(url, data=payload, headers=headers)
        yield res.read()


def get_agentartificial_generator():
    """
    Returns an instance of the AgentArtificialGenerator class.

    This function creates and returns an instance of the AgentArtificialGenerator class. The instance can be used to generate text using the AgentArtificial API.

    Returns:
        AgentArtificialGenerator: An instance of the AgentArtificialGenerator class.
    """
    return AgentArtificialGenerator()


def get_agentartficial_generator():
    return AgentArtificialGenerator()


available_generators.add_generator("agent_artificial", get_agentartificial_generator)

      ```

    - 📄 **AnthropicGenerator.py**

      📄 *File Path*: `./src/text_generators/AnthropicGenerator.py`

      ```
      import os
from typing import List, Dict, Optional, AsyncGenerator
from anthropic import AsyncAnthropic
from src.text_generators.interface import Generator, available_generators
from src.templates.handler import 


TEMPLATES = get_template_manager()


class AnthropicGenerator(Generator):
    def __init__(self):
        super().__init__(
            templates=
            url=os.getenv("ANTHROPIC_URL", "https://api.anthropic.com/v1/messages"),
            api=os.getenv("ANTHROPIC_API_KEY"),
            context_window=100000,
            context=[],
        )
        self.client = AsyncAnthropic(api_key=self.api)
        self.model = "claude-3-opus-20240229"
        self.max_tokens = 1024

    async def generate(self, queries: List[str], context: List[str]) -> str:
        messages = self.prepare_messages(queries, context)
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=messages
            )
            return response.content[0].text
        except Exception as e:
            raise Exception(f"Error in AnthropicGenerator.generate: {e}")

    async def generate_stream(self, queries: List[str], context: List[str]) -> AsyncGenerator[Dict[str, Optional[str]], None]:
        messages = self.prepare_messages(queries, context)
        
        try:
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=messages
            ) as stream:
                async for response in stream:
                    if response.type == "content_block_delta":
                        yield {
                            "message": response.delta.text,
                            "finish_reason": None,
                        }
                yield {
                    "message": "",
                    "finish_reason": "stop",
                }
        except Exception as e:
            raise Exception(f"Error in AnthropicGenerator.generate_stream: {e}")

    def prepare_messages(self, queries: List[str], context: List[str]) -> List[Dict[str, str]]:
        messages = []
        
        if self.template:
            messages.append({"role": "system", "content": self.template.create_system_prompt()})
        
        for ctx in context:
            messages.append({"role": "user", "content": ctx})
            messages.append({"role": "assistant", "content": "Understood."})
        
        query = " ".join(queries)
        messages.append({"role": "user", "content": query})
        
        return messages

    def set_model(self, model: str) -> None:
        self.model = model

    def set_max_tokens(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens

    # Implementing abstract methods
    def set_apikey(self) -> bool:
        if self.api:
            self.client = AsyncAnthropic(api_key=self.api)
            return True
        return False

    def set_url(self) -> bool:
        # Anthropic API URL is set by default in the AsyncAnthropic client
        return True

    def set_context(self) -> bool:
        # Context is managed in the prepare_messages method
        return True

    def set_context_window(self) -> bool:
        # Context window is not directly used in the Anthropic API
        return True

    def set_template(self) -> bool:
        if self.template:
            return True
        return False

def get_anthropic_generator():
    return AnthropicGenerator()

available_generators.add_generator("anthropic", get_anthropic_generator)
      ```

    - 📄 **ChatGPT4Generator.py**

      📄 *File Path*: `./src/text_generators/ChatGPT4Generator.py`

      ```
      import contextlib
import os
from loguru import logger
from openai import OpenAI
from typing import List, Dict, AsyncGenerator
from src.text_generators.interface import Generator, GPT4GeneratorError

openai = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), 
    base_url=os.getenv("OPENAI_BASE_URL")
)

class GPT4Generator(Generator):
    """
    GPT4 Generator.
    """

    def __init__(self):
        super().__init__()
        self.name = "GPT4Generator"
        self.generator = openai
        self.description = "Generator using OpenAI's GPT-4-1106-preview model"
        self.requires_library = ["openai"]
        self.requires_env = ["OPENAI_API_KEY"]
        self.streamable = True
        self.model = os.getenv("OPENAI_MODEL")
        self.context_window = 10000
        self.context = {}
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")

    async def generate(
        self,
        queries: List[str],
        context: List[str],
    ) -> str:
        self.set_context(context)
        messages = self.prepare_messages(queries, self.context)

        try:
            completion = openai.chat.completions.create(messages, model=self.model_name)
            system_msg = str(completion["choices"][0]["message"]["content"])

        except GPT4GeneratorError as e:
            logger.error(f"Error in GPT4Generator.generate: {e}")
        return system_msg

    async def generate_stream(
        self,
        queries: List[str],
        context: List[str] = None
    ) -> AsyncGenerator[Dict]:
        self.set_context(context)
        messages = self.prepare_messages(queries, self.context)

        try:
            completion = openai.chat.completions.create(messages, streaming=True)

            with contextlib.suppress(StopAsyncIteration):
                while True:
                    chunk = completion.__anext__()
                    if "content" in chunk["choices"][0]["delta"]:
                        yield {
                            "message": chunk["choices"][0]["delta"]["content"],
                            "finish_reason": chunk["choices"][0]["finish_reason"],
                        }
                    else:
                        yield {
                            "message": "",
                            "finish_reason": chunk["choices"][0]["finish_reason"],
                        }
        except GPT4GeneratorError as e:
            logger.error(f"Error in GPT4Generator.generate_stream {e}")

    def prepare_messages(
        self, queries: List[str], context: List[str]
    ) -> Dict[str, str]:
        messages.extend([message for message in self.context])
        query = " ".join(queries)
        user_context = " ".join(context)

        messages.append(
            {
                "role": "user",
                "content": f"Please answer this query: '{query}' with this provided context: {user_context}",
            }
        )

        return messages

    def set_apikey(self, api_key=None) -> bool:
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        self.api_key = api_key
        self.set_generator()
        return bool(self.api_key)

    def set_url(self, url=None) -> bool:
        if url is None:
            url = os.getenv("OPENAI_URL")
        self.base_url = url
        self.set_generator()
        return bool(self.base_url)
        
    def set_generator(self, generator=None):
        if generator is None:
            generator = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        self.generator = generator
        return bool(self.generator)
        
    def set_model(self, model=None):
        if model is None:
            model = os.getenv("OPENAI_MODEL")
        self.model = model
        return bool(self.model)
        
    def set_context(self, context=None) -> bool:
        if context is None:
            context = {}
        self.context = context
        return bool(self.context)

    def set_context_window(self, context_window=None) -> bool:
        if self.context_window is None:
            context_window = 16000
        self.context_window = context_window
        return bool(self.context_window)


def get_GPT4Generator():
    return GPT4Generator

      ```

    - 📄 **HuggingFaceGenerator.py**

      📄 *File Path*: `./src/text_generators/HuggingFaceGenerator.py`

      ```
      from src.text_generators.interface import Generator, available_generators
from src.templates.interface import AvailableTemplates


class Llama2Generator(Generator):
    """
    Llama2Generator Generator.
    """

    def __init__(self, selected_model: str):
        super().__init__(
            api=os.getenv("HUGGINGFACE_API_KEY"),
            url=f"https://huggingface.co/{selected_model}",
            template=available_templates.templates[selected_model],
            context_window=8000,
            context=[],
        )
        self.name = "Llama2Generator"
        self.description = "Generator using Meta's Llama-2-7b-chat-hf model"
        self.streamable = True
        self.model = None
        self.tokenizer = None
        self.device = None
        self.context_window = 3000
        if os.environ.get("HF_TOKEN", ""):
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                def get_device():
                    if torch.cuda.is_available():
                        return torch.device("cuda")
                    elif torch.backends.mps.is_available():
                        return torch.device("mps")
                    else:
                        return torch.device("cpu")

                self.device = get_device()

                self.model = AutoModelForCausalLM.from_pretrained(
                    "meta-llama/Llama-2-7b-chat-hf",
                    device_map=self.device,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "meta-llama/Llama-2-7b-chat-hf",
                    device_map=self.device,
                )
                self.model = self.model.to(self.device)
                msg.info("Loading Llama Model")
            except Exception as e:
                msg.warn(str(e))

    async def generate_stream(
        self,
        queries: List[str],
        context: List[str],
        conversation: Dict = None,
    ) -> Iterator[Dict]:
        """Generate a stream of response Dicts based on a List of queries and List of contexts, and includes conversational context
        @parameter: queries : List[str] - List of queries
        @parameter: context : List[str] - List of contexts
        @parameter: conversation : Dict - Conversational context
        @returns Iterator[Dict] - Token response generated by the Generator in this format {system:TOKEN, finish_reason:stop or empty}.
        """
        if conversation is None:
            conversation = {}
        messages = self.prepare_messages(queries, context, conversation)

        try:
            import torch

            prompt = messages
            output_length = 500

            # Ensure the tokenizer has a padding token defined
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            input = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=3500,  # Specify a max_length value
            )

            input = {k: v.to(self.device) for k, v in input.items()}
            attention_mask = input.get(
                "attention_mask", None
            )  # Get the attention mask, if provided
            input_len = input["input_ids"].shape[1]
            msg.info(f"Tokenized finished with {input_len} tokens")

            position_ids = torch.arange(
                0, input_len, dtype=torch.long, device=self.device
            ).unsqueeze(0)

            for output_len in range(output_length):
                updated_position_ids = torch.arange(
                    0, input_len + output_len, dtype=torch.long, device=self.device
                ).unsqueeze(0)
                output = await asyncio.to_thread(
                    lambda: self.model.generate(
                        input_ids=input["input_ids"],
                        attention_mask=attention_mask,
                        position_ids=updated_position_ids,
                        max_length=input_len + output_len + 1,
                        do_sample=True,
                        temperature=0.1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                )
                current_token_id = output[0][-1]
                if current_token_id == self.tokenizer.eos_token_id:
                    break
                current_token = self.tokenizer.convert_ids_to_tokens(
                    [current_token_id], skip_special_tokens=False
                )
                if type(current_token) == List:
                    current_token = " ".join(current_token)
                current_token = current_token.replace("<0x0A>", "\n").replace("▁", " ")
                # Update input for next iteration
                input["input_ids"] = torch.cat(
                    (input["input_ids"], current_token_id.view(1, 1)), dim=1
                )
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones((1, 1), dtype=torch.long, device=self.device),
                    ),
                    dim=1,
                )  # Update the attention_mask
                position_ids = torch.cat(
                    (
                        position_ids,
                        torch.tensor([[input_len + output_len]], device=self.device),
                    ),
                    dim=1,
                )
                yield {
                    "message": current_token,
                    "finish_reason": "",
                }
            yield {
                "message": "",
                "finish_reason": "stop",
            }

        except Exception:
            raise

    def prepare_messages(self, queries, context, conversation):
        """
        Prepares a List of messages formatted for a Retrieval Augmented Generation chatbot system, including system instructions, previous conversation, and a new user query with context.

        @parameter queries: A List of strings representing the user queries to be answered.
        @parameter context: A List of strings representing the context information provided for the queries.
        @parameter conversation: A List of previous conversation messages that include the role and content.

        @returns A List or of message Dictionaries or whole prompts formatted for the chatbot. This includes an initial system message, the previous conversation messages, and the new user query encapsulated with the provided context.

        Each message in the List is a Dictionary with 'role' and 'content' keys, where 'role' is either 'system' or 'user', and 'content' contains the relevant text. This will depend on the LLM used.
        """
        llama_prompt = """
        <s>[INST] <<SYS>>
        You are a Retrieval Augmented Generation chatbot. Answer user queries using only the provided context. If the context does not provide enough information, say so. If the answer requires code examples encapsulate them with ```programming-language-name ```. Don't do pseudo-code. \n<</SYS>>\n\n
        """

        query = " ".join(queries)
        user_context = " ".join(context)

        llama_prompt += (
            f"Answer this query: '{query}' with this context: {user_context} [/INST] "
        )

        return llama_prompt


def get_huggingface_generator(api_key: str):
    return HuggingFaceGenerator(api_key=api_key)


available_generators.generators["huggingface"] = get_huggingface_generator

      ```

    - 📄 **interface.py**

      📄 *File Path*: `./src/text_generators/interface.py`

      ```
      from pydantic import BaseModel
from typing_extensions import List, Dict, Iterator, Callable
from abc import ABC, abstractmethod


class GeneratorError(Exception):
    """Base class for all Generator errors."""
    
    
class GPT4GeneratorError(GeneratorError):
    """Error raised when an error occurs in the GPT4Generator."""
    

class Generator(BaseModel, ABC):
    template: Callable
    url: str
    api: str
    context: List
    context_window: int

    @abstractmethod
    def set_apikey(self) -> bool:
        """
        Sets the API key for the Generator.

        This method is responsible for setting the API key for the Generator. It should be implemented by any class that inherits from the Generator class.

        Returns:
            bool: True if the API key is successfully set, False otherwise.
        """

    @abstractmethod
    def set_url(self) -> bool:
        """
        Sets the URL for the current object.

        :return: A boolean indicating whether the URL was successfully set.
        """

    @abstractmethod
    def set_context(self) -> bool:
        """
        Sets the context for the current object.

        This method is responsible for setting the context for the current object. It should be implemented by any class that inherits from the current class.

        Returns:
            bool: True if the context is successfully set, False otherwise.
        """

    @abstractmethod
    def set_context_window(self) -> bool:
        """
        Sets the context window for the current object.

        This method is responsible for setting the context window for the current object. It should be implemented by any class that inherits from the current class.

        Returns:
            bool: True if the context window is successfully set, False otherwise.
        """

    @abstractmethod
    def set_template(self) -> bool:
        """
        A description of the entire function, its parameters, and its return types.
        """

    @abstractmethod
    async def generate(self, messages: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Asynchronously generates a response based on a list of messages.

        Args:
            messages (List[Dict[str, str]]): A list of dictionaries representing messages.

        Returns:
            Dict[str, str]: A dictionary representing the generated response.
        """

    @abstractmethod
    async def generate_stream(
        self, messages: List[Dict[str, str]]
    ) -> Iterator[Dict[str, str]]:
        """
        Asynchronously generates a stream of response dictionaries based on a list of messages.

        Args:
            messages (List[Dict[str, str]]): A list of dictionaries representing messages.

        Yields:
            Iterator[Dict[str, str]]: An iterator that yields dictionaries representing the generated responses.
        """

    @abstractmethod
    def prepare_messages(
        self,
        queries: List[str],
        context: List[Dict[str, str]],
        messages: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        A description of the entire function, its parameters, and its return types.
        """


class GeneratorConfig(BaseModel):
    template: Callable
    url: str
    api: str
    context: List
    context_window: int


class AvailableGenerators(BaseModel):
    generators: Dict[str, Callable] = {}

    def add_generator(self, name: str, generator):
        """
        Add a new generator to the AvailableGenerators instance.

        Args:
            name (str): The name of the generator to add.
            generator (Callable): The generator function to add.

        Returns:
            None
        """
        self.generators[name] = generator

    def remove_generator(self, name: str):
        """
        Remove a generator from the AvailableGenerators instance.

        Args:
            name (str): The name of the generator to remove.

        Returns:
            None
        """
        del self.generators[name]

    def get_generator(self, name: str) -> Callable:
        """
        Retrieves a generator function by name.

        Args:
            name (str): The name of the generator to retrieve.

        Returns:
            Callable: The generator function associated with the provided name.
        """
        return self.generators[name]

    def get_all_generators(self) -> Dict[str, Callable]:
        """
        Returns a dictionary containing all the generators available.

        :return: A dictionary where the keys are the names of the generators and the values are the callable functions.
        :rtype: Dict[str, Callable]
        """
        return self.generators

    def get_generator_names(self) -> List[str]:
        """
        Returns a list of names of all available generators.

        :return: A list of strings representing the names of the generators.
        :rtype: List[str]
        """
        return list(self.generators.keys())

    def get_generator_by_name(self, name: str) -> Generator:
        """
        Retrieves a generator function by name.

        Args:
            name (str): The name of the generator to retrieve.

        Returns:
            Generator: The generator function associated with the provided name.
        """
        return self.generators[name]()


available_generators = AvailableGenerators()

      ```

    - 📄 **manager.py**

      📄 *File Path*: `./src/text_generators/manager.py`

      ```
      from src.text_generators.interface import Generator, available_generators
from typing import Dict, List


all_generators = available_generators.get_all_generators()


class GeneratorManager:
    def __init__(self, selected_generator: str):
        """
        Initializes a new instance of the GeneratorManager class.

        Args:
            selected_generator (str): The name of the selected generator.

        Returns:
            None
        """
        self.selected_generator = selected_generator
        self.generators = {**all_generators}
        self.accumulated_tokens = 0
        self.truncated_dicts = []
        self.generator = self.set_generator(selected_generator)

    def set_generator(self, generator_name: str) -> Generator:
        """
        Sets the generator based on the provided generator_name.

        Args:
            generator_name (str): The name of the generator to set.

        Returns:
            Generator: The generator that is set.
        """
        print(self.generators)
        self.generator = self.generators[generator_name]
        return self.generator

    def list_generators(self) -> List[str]:
        """
        Returns a list of generator names.

        :return: A list of strings representing the names of the available generators.
        :rtype: List[str]
        """
        return list(self.generators.keys())

    def truncate_context_dicts(
        self, context_dicts: List[Dict[str, str]], max_tokens: int
    ) -> List[Dict[str, str]]:
        """
        Truncates a list of conversation dictionaries to a maximum number of tokens.

        Args:
            context_dicts (List[Dict[str, str]]): A list of dictionaries representing conversations,
                where each dictionary has keys "type", "content", and "typewriter".
            max_tokens (int): The maximum number of tokens allowed for the truncated conversations.

        Returns:
            List[Dict[str, str]]: A list of dictionaries representing truncated conversations.
                Each dictionary has keys "type", "content", and "typewriter".

        """
        import tiktoken

        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

        # Start with the newest conversations
        for item_dict in reversed(context_dicts):
            item_tokens = encoding.encode(item_dict["content"], disallowed_special=())

            # If adding the entire new item exceeds the max tokens
            if self.accumulated_tokens + len(item_tokens) > max_tokens:
                # Calculate how many tokens we can add from this item
                remaining_space = max_tokens - self.accumulated_tokens
                truncated_content = encoding.decode(item_tokens[:remaining_space])

                # Create a new truncated item Dictionary
                truncated_item_dict = {
                    "type": item_dict["type"],
                    "content": truncated_content,
                    "typewriter": item_dict["typewriter"],
                }

                self.truncated_dicts.append(truncated_item_dict)
                break

            self.truncated_dicts.append(item_dict)
            self.accumulated_tokens += len(item_tokens)

        # The List has been built in reverse order so we reverse it again
        return list(reversed(self.truncated_dicts))

      ```

  - 📁 **ui/**
    - 📄 **chat_widget.py**

      📄 *File Path*: `./src/ui/chat_widget.py`

      ```
      from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QComboBox, QHBoxLayout
from PyQt5.QtCore import pyqtSignal
from custom_text_edit import CustomTextEdit

class ChatWidget(QDialog):
    """
    ChatWidget is a dialog widget for managing chat interactions.

    This class provides methods to initialize the UI components, handle user input, and manage
    chat history within the dialog.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        # Initialize the UI components
        pass

    def create_widgets(self):
        # Create widgets
        pass

    def set_widget_properties(self):
        # Set properties for widgets
        pass

    def create_widget_layouts(self):
        # Create layouts for widgets
        pass

    def set_widget_connections(self):
        # Setup connections for widgets
        pass

    def on_combobox_changed(self, index):
        # Handle combobox change
        pass

    def create_chat_history(self):
        # Create chat history
        pass

    def create_user_input(self):
        # Create user input area
        pass

    def adjust_user_input_height(self):
        # Adjust the height of the user input
        pass

    def call_send_message(self, user_message):
        # Call send message
        pass

    def send_message(self, user_message):
        # Send message
        pass

    def open_large_text_input(self):
        # Open large text input dialog
        pass

    def clear_chat_history(self):
        # Clear chat history
        pass

    def open_file_dialog(self):
        # Open file dialog
        pass

    def set_chat_message(self, file_dialog):
        # Set chat message
        pass

    def run_command(self, text):
        # Run command
        pass

    def display_help(self):
        # Display help
        pass

    def load_chat_history(self):
        # Load chat history
        pass

    def save_chat_history(self):
        # Save chat history
        pass

    def set_chat_style(self):
        # Set chat style
        pass

    def exit(self):
        # Exit chat widget
        pass

      ```

    - 📄 **custom_text_edit.py**

      📄 *File Path*: `./src/ui/custom_text_edit.py`

      ```
      from PyQt5.QtWidgets import QTextEdit

class CustomTextEdit(QTextEdit):
    def __init__(self, *args, **kwargs):
        super(CustomTextEdit, self).__init__(*args, **kwargs)

    # Handling key events
    def keyPressEvent(self, event):
        # Custom key event handling logic here
        pass

      ```

    - 📄 **custom_title_bar.py**

      📄 *File Path*: `./src/ui/custom_title_bar.py`

      ```
      from PyQt5.QtWidgets import QLabel, QHBoxLayout, QWidget
from PyQt5.QtCore import Qt

class CustomTitleBar(QWidget):
    """
    CustomTitleBar is a widget that provides a custom title bar for the application.

    This class manages the appearance and functionality of the title bar, including the title label
    and any additional buttons.
    """

    def __init__(self, parent=None):
        super(CustomTitleBar, self).__init__(parent)
        self.setFixedHeight(30)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        # Add title label
        self.titleLabel = QLabel(self)
        layout.addWidget(self.titleLabel)
        self.titleLabel.setObjectName("HexAmerous")
        self.titleLabel.setText("HexAmerous")
        self.titleLabel.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.titleLabel.setFixedHeight(30)
        layout.addWidget(self.buttons())

        # Set stylesheet
        self.setStyleSheet(
            """
            QPushButton {
                border: none;
                background-color: rgba(67, 3, 81, 0.3);
                color: #f9f9f9;
            }
            QPushButton:hover {
                background-color: rgba(67, 3, 81, 0.6);;
                color: #f9f9f9;
            }
            QPushButton:pressed {
                background-color: #430351;
                color: #f9f9f9;
            }
            QLabel {
                background-color: rgba(67, 3, 81, 0.9);
                color: #f9f9f9;
                font-size: 20pt;
                font-weight: bold;
                text-align: center;
                font-family: 'Cascadia Code';

            }
        """
        )

    def buttons(self):
        # Create buttons
        pass

      ```

    - 📄 **large_text_input_dialog.py**

      📄 *File Path*: `./src/ui/large_text_input_dialog.py`

      ```
      from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton
from PyQt5.QtCore import Qt

class LargeTextInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Large Text Input")
        self.resize(600, 900)
        self.text_input = QTextEdit()
        self.text_input.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.text_input.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.text_input.setStyleSheet(
            "background-color: rgba(67, 3, 81, 0.4); color: #f9f9f9; font-family 'Cascadia Code'; font-size: 12pt; font-weight: bold;"
        )
        self.send_button = QPushButton("Send")
        self.send_button.setStyleSheet(
            "background-color:rgba(67, 3, 81, 0.4); color: #f9f9f9; font-family 'Cascadia Code'; font-size: 14pt; font-weight: bold;"
        )
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.text_input)
        self.layout.addWidget(self.send_button)
        self.send_button.clicked.connect(self.send_large_text)

    def send_large_text(self):
        """
        Sends the large text input to a designated handler.

        This method is triggered when the send button is clicked. It handles the logic for processing
        and sending the text input from the dialog.
        """
        # Logic to send large text
        pass

      ```

    - 📄 **main_window.py**

      📄 *File Path*: `./src/ui/main_window.py`

      ```
      from PyQt5.QtWidgets import QVBoxLayout, QSizeGrip
from PyQt5.QtCore import Qt
from custom_title_bar import CustomTitleBar
from chat_widget import ChatWidget

class MainWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.resize(728, 1024)
        self.flags = Qt.WindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        self.setWindowFlags(self.flags)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.size_grip = QSizeGrip(self)
        self.size_grip.setStyleSheet("width: 10px; height: 5px; margin 0px;")
        self.layout.addWidget(self.size_grip)
        self.titleBar = CustomTitleBar(self)
        self.layout.addWidget(self.titleBar)
        self.chat_widget = ChatWidget()
        self.layout.addWidget(self.chat_widget)

        self.image = ""
        self.background = self.change_background_image()

    def change_background_image(self, image="./imgs/00004.png"):
        # Change background image logic
        pass

      ```

    - 📄 **scroll_area.py**

      📄 *File Path*: `./src/ui/scroll_area.py`

      ```
      from PyQt5.QtWidgets import QFrame, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt

class ScrollArea(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("background-color: transparent; border: 0px;")
        self.setWidgetResizable(True)
        self.content_widget = QWidget()
        self.content_widget.setAutoFillBackground(False)
        self.content_widget_layout = QVBoxLayout()
        self.content_widget.setLayout(self.content_widget_layout)
        self.setWidget(self.content_widget)

      ```

    - 📄 **signal.py**

      📄 *File Path*: `./src/ui/signal.py`

      ```
      from PyQt5.QtCore import QObject, pyqtSignal

class Signal(QObject):
    close_signal = pyqtSignal(name="close_signal")

    def emit(self):
        self.close_signal.emit()

      ```

  - 📁 **utils/**
    - 📄 **__init__.py**

      📄 *File Path*: `./src/utils/__init__.py`

      ```
      
      ```

    - 📄 **coala.py**

      📄 *File Path*: `./src/utils/coala.py`

      ```
      from typing import Literal
from openai.pagination import SyncCursorPage
from data_models import run
from openai.types.beta.threads import ThreadMessage
from openai.types.beta.threads.runs import RetrievalToolCall
from utils.tools import Actions
from utils.tools import ActionItem


class CoALA:
    def __init__(
        self,
        runsteps: SyncCursorPage[run.RunStep],
        messages: SyncCursorPage[ThreadMessage],
        job_summary: str,
        tools_map: dict[str, ActionItem],
    ):
        """
        CoALA class to setup the CoALA prompt
        messages: episodic memory
        runsteps: working memory
        job_summary: objective
        tools_map: external actions
        """
        self.runsteps = runsteps
        self.messages = messages
        self.job_summary = job_summary
        self.tools_map = tools_map

    def compose_trace(self):
        """
        Compose the trace prompt of the current task
        """
        trace_prompt = []
        for step in self.runsteps:
            if step.type == "tool_calls":
                trace_prompt.append(f"Action: {step.step_details.tool_calls[0].type}")
                trace_prompt.append(
                    f"Observation: {step.step_details.tool_calls[0].model_dump()}"
                )
            if step.type == "message_creation":
                message = next(
                    (
                        msg.content[0].text.value
                        for msg in self.messages.data
                        if msg.id == step.step_details.message_creation.message_id
                    ),
                    None,
                )
                trace_prompt.append(f"Thought: {message}")
        return "\n".join(trace_prompt)

    def compose_actions(self):
        """
        Compose the tools prompt for the CoALA task
        """
        action_prompts = []
        for tool in self.tools_map:
            action_prompts.append(f"- {tool} ({self.tools_map[tool].description})")
        action_prompts.append(
            f"- {Actions.COMPLETION.value} (Finish the process, generate the final answer)"
        )
        return "\n".join(action_prompts)

    def compose_prompt(self, type: Literal["action", "thought", "final_answer"]):
        """
        Compose the prompt for the CoALA task
        """
        base_prompt = None
        if type == "action":
            base_prompt = """Your role is to determine which "Action" to use next.
You must always begin with "Action: ..." """
        elif type == "thought":
            base_prompt = """Your role is to provide a "Thought" response to the user.
You must always begin with "Thought: ..."  and finish with "Action: " """
        elif type == "final_answer":
            base_prompt = """Your role is to provide the "Final Answer" to the user.
You must always begin with "Final Answer: ..." """
        trace_prompt = self.compose_trace()
        actions_prompt = self.compose_actions()
        actions_list = list(self.tools_map.keys()) + [Actions.COMPLETION.value]

        coala_prompt = f"""{base_prompt}
You will observe that there are already steps after "Begin!".
The actions available to you are:

{actions_prompt}

Continue the generation.
Only reply with the single next step.
Do respond with more than the immediate next step.
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of {actions_list}
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {self.job_summary}
{trace_prompt}"""

        return coala_prompt

      ```

    - 📄 **logging.py**

      📄 *File Path*: `./src/utils/logging.py`

      ```
      import os
from loguru import logger
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Logger:
    def __init__(self, level: str = "INFO"):
        self.logger = logger
        self.level = level

    def log(self, message: str, level: Optional[str] = None):
        if level is None:
            level = self.level
        if level == "INFO":
            self.logger.info(message)
        elif self.level == "DEBUG":
            self.logger.debug(message)
        elif self.level == "WARNING":
            self.logger.warning(message)
        elif self.level == "CRITICAL":
            self.logger.critical(message)
        elif self.level == "SUCCESS":
            self.logger.success(message)
        elif self.level == "TRACE":
            self.logger.trace(message)
        elif self.level == "EXCEPTION":
            self.logger.exception(message)

    def log_info(self, message: str):
        self.logger.info(message)

    def log_error(self, message: str):
        self.logger.error(message)


    def log_debug(self, message: str):
        self.logger.debug(message)


    def log_warning(self, message: str):
        self.logger.warning(message)


    def log_critical(self, message: str):
        self.logger.critical(message)


    def log_success(self, message: str):
        self.logger.success(message)


    def log_trace(self, message: str):
        self.logger.trace(message)


    def log_exception(self, message: str):
        self.logger.exception(message)
        
    def get_logger(self):
        return self.logger

def get_logger(level: str = None):        
    log_level = level.upper() if level else os.getenv("LOG_LEVEL", "INFO")
    return Logger(log_level).get_logger()

if __name__ == "__main__":
    log = get_logger()
    log.info("Hello World")

      ```

    - 📄 **openai_clients.py**

      📄 *File Path*: `./src/utils/openai_clients.py`

      ```
      from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)

litellm_client = None
if os.getenv("LITELLM_API_URL"):
    litellm_client = OpenAI(
        api_key=os.getenv("LITELLM_API_KEY"),
        base_url=os.getenv("LITELLM_API_URL", None),
    )
else:
    litellm_client = OpenAI(
        api_key=os.getenv("LITELLM_API_KEY"),
    )

assistants_client = OpenAI(
    base_url=os.getenv("ASSISTANTS_API_URL"),
)

      ```

    - 📄 **ops_api_handler.py**

      📄 *File Path*: `./src/utils/ops_api_handler.py`

      ```
      # api_handler.py
from typing import List, Literal
import requests
import os
from dotenv import load_dotenv
from data_models import run
from openai.types.beta.threads import ThreadMessage
from openai.types.beta.threads.runs import RetrievalToolCall
from openai.types.beta.threads.runs.web_retrieval_tool_call import WebRetrievalToolCall
from utils.openai_clients import assistants_client

# TODO: create run script that imports env vars
load_dotenv()
BASE_URL = os.getenv("ASSISTANTS_API_URL")


def update_run(thread_id: str, run_id: str, run_update: run.RunUpdate) -> run.Run:
    """
    Update the status of a Run.

    Parameters:
    thread_id (str): The ID of the thread.
    run_id (str): The ID of the run.
    new_status (str): The new status to set for the run.

    Returns:
    bool: True if the status was successfully updated, False otherwise.
    """
    update_url = f"{BASE_URL}/ops/threads/{thread_id}/runs/{run_id}"
    update_data = run_update.model_dump(exclude_none=True)

    response = requests.post(update_url, json=update_data)

    if response.status_code == 200:
        return run.Run(**response.json())
    else:
        return None


def create_message(
    thread_id: str, content: str, role: Literal["user", "assistant"]
) -> ThreadMessage:
    # Create a thread with a message
    message = assistants_client.beta.threads.messages.create(
        thread_id=thread_id, content=content, role=role
    )
    assert message.thread_id == thread_id

    return message


def create_message_runstep(
    thread_id: str, run_id: str, assistant_id: str, content: str
) -> run.RunStep:
    message = create_message(thread_id, content, role="assistant")
    # Prepare run step details
    run_step_details = {
        "assistant_id": assistant_id,
        "step_details": {
            "type": "message_creation",
            "message_creation": {"message_id": message.id},
        },
        "type": "message_creation",
        "status": "completed",
    }
    run_step_details = run.RunStepCreate(**run_step_details).model_dump(
        exclude_none=True
    )

    # Post request to create a run step
    response = requests.post(
        f"{BASE_URL}/ops/threads/{thread_id}/runs/{run_id}/steps", json=run_step_details
    )
    if response.status_code != 200:
        raise Exception(f"Failed to create run step: {response.text}")

    return run.RunStep(**response.json())


def create_retrieval_runstep(
    thread_id: str, run_id: str, assistant_id: str, documents: List[str]
) -> dict:
    # Assuming the `ToolCall` is properly defined elsewhere to include `RetrievalToolCall`.
    tool_call = RetrievalToolCall(
        id="unique_tool_call_id",  # This should be a unique identifier.
        retrieval={"documents": documents},
        type="retrieval",
    )

    # Prepare run step details with the tool call
    run_step_details = {
        "assistant_id": assistant_id,
        "step_details": {
            "type": "tool_calls",
            "tool_calls": [tool_call.model_dump()],  # Serialize `ToolCall` to a dict
        },
        "type": "tool_calls",
        "status": "completed",
    }

    # This model dumping part would be dependent on how you're handling Pydantic models, showing a conceptual example:
    run_step_details = run.RunStepCreate(**run_step_details).model_dump(
        exclude_none=True
    )

    # Post request to create a run step
    response = requests.post(
        f"{BASE_URL}/ops/threads/{thread_id}/runs/{run_id}/steps", json=run_step_details
    )
    if response.status_code != 200:
        raise Exception(f"Failed to create run step: {response.text}")

    return run.RunStep(**response.json())


def create_web_retrieval_runstep(
    thread_id: str, run_id: str, assistant_id: str, documents: List[str], site: str
) -> dict:
    # Assuming the `ToolCall` is properly defined elsewhere to include `RetrievalToolCall`.
    tool_call = WebRetrievalToolCall(
        id="unique_tool_call_id",  # TODO: This should be a unique identifier.
        retrieval={"documents": documents},
        site=site,
        type="web_retrieval",
    )

    # Prepare run step details with the tool call
    run_step_details = {
        "assistant_id": assistant_id,
        "step_details": {
            "type": "tool_calls",
            "tool_calls": [tool_call.model_dump()],  # Serialize `ToolCall` to a dict
        },
        "type": "tool_calls",
        "status": "completed",
    }

    # This model dumping part would be dependent on how you're handling Pydantic models, showing a conceptual example:
    run_step_details = run.RunStepCreate(**run_step_details).model_dump(
        exclude_none=True
    )

    # Post request to create a run step
    response = requests.post(
        f"{BASE_URL}/ops/threads/{thread_id}/runs/{run_id}/steps", json=run_step_details
    )
    if response.status_code != 200:
        raise Exception(f"Failed to create run step: {response.text}")

    return run.RunStep(**response.json())

      ```

    - 📄 **tools.py**

      📄 *File Path*: `./src/utils/tools.py`

      ```
      from enum import Enum
from typing import List
from openai.types.beta.assistant import (
    Tool,
    ToolWebRetrieval,
    ToolRetrieval,
    ToolFunction,
    ToolCodeInterpreter,
)
from pydantic import BaseModel


class Actions(Enum):
    # function, retrieval, code_interpreter, text_generation, completion
    FUNCTION = "function"
    RETRIEVAL = "retrieval"
    WEB_RETREIVAL = "web_retrieval"
    CODE_INTERPRETER = "code_interpreter"
    TEXT_GENERATION = "text_generation"
    COMPLETION = "completion"
    FAILURE = "failure"


Actions["FAILURE"]


class ActionItem(BaseModel):
    type: str
    description: str


def actions_to_map(actions: List[str]) -> dict[str, ActionItem]:
    """
    Converts a list of AssistantTool objects to a dictionary.
    """
    actions_map = {}
    for action in actions:
        if action == Actions.TEXT_GENERATION.value:
            actions_map[action] = ActionItem(
                type=Actions.TEXT_GENERATION.value,
                description="Communicate to the user either to summarize or express the next tasks to be executed.",
            )
        elif action == Actions.COMPLETION.value:
            actions_map[action] = ActionItem(
                type=Actions.COMPLETION.value,
                description="Indicate that the current goal is achieved or that this loop should be terminated.",
            )
    return actions_map


def tools_to_map(tools: List[Tool]) -> dict[str, ActionItem]:
    """
    Converts a list of AssistantTool objects to a dictionary.
    """
    tools_map = {}
    for tool in tools:
        if isinstance(tool, ToolFunction):
            tools_map[tool.type] = ActionItem(
                type=tool.type,
                description="Generates text based on input data.",
            )
        elif isinstance(tool, ToolRetrieval):
            tools_map[tool.type] = ActionItem(
                type=tool.type,
                description="Retrieves information from files provided.",
            )
        elif isinstance(tool, ToolWebRetrieval):
            tools_map[tool.type] = ActionItem(
                type=tool.type,
                description="Retrieves information from the web.",  # TODO: Sean, please provide a better description.
            )
        elif isinstance(tool, ToolCodeInterpreter):
            tools_map[tool.type] = ActionItem(
                type=tool.type,
                description="Interprets and executes code.",
            )
    return tools_map


text_generation_tool = ActionItem(
    type="text_generation",
    description="General text response.",
)

completion_tool = ActionItem(
    type="completion",
    description="Completes the task.",
)

      ```

    - 📄 **weaviate_utils.py**

      📄 *File Path*: `./src/utils/weaviate_utils.py`

      ```
      import weaviate
from typing import List, Dict
import os

client = weaviate.Client(
    url=os.getenv("WEAVIATE_URL"),
    auth_client_secret=weaviate.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY"))
)

def store_document(document: Dict[str, str]) -> str:
    """
    Store a document in Weaviate.
    
    Args:
    document (Dict[str, str]): A dictionary containing the document data.
    
    Returns:
    str: The UUID of the stored document.
    """
    return client.data_object.create(
        data_object=document,
        class_name="Document"
    )

def retrieve_documents(query: str, limit: int = 5) -> List[Dict[str, str]]:
    """
    Retrieve documents from Weaviate based on a semantic search query.
    
    Args:
    query (str): The search query.
    limit (int): The maximum number of documents to retrieve.
    
    Returns:
    List[Dict[str, str]]: A list of retrieved documents.
    """
    result = (
        client.query
        .get("Document", ["content", "metadata"])
        .with_near_text({"concepts": [query]})
        .with_limit(limit)
        .do()
    )
    return result["data"]["Get"]["Document"]

def retrieve_file_chunks(file_ids: List[str], query: str, limit: int = 5) -> List[Dict[str, str]]:
    """
    Retrieve file chunks from Weaviate based on file IDs and a semantic search query.
    
    Args:
    file_ids (List[str]): List of file IDs to search within.
    query (str): The search query.
    limit (int): The maximum number of chunks to retrieve.
    
    Returns:
    List[Dict[str, str]]: A list of retrieved file chunks.
    """
    result = (
        client.query
        .get("FileChunk", ["content", "metadata"])
        .with_near_text({"concepts": [query]})
        .with_where({
            "path": ["metadata", "file_id"],
            "operator": "In",
            "valueString": file_ids
        })
        .with_limit(limit)
        .do()
    )
    return result["data"]["Get"]["FileChunk"]

      ```

  - 📁 **vectordb/**
    - 📄 **VectorDB.py**

      📄 *File Path*: `./src/vectordb/VectorDB.py`

      ```
      
      ```

    - 📄 **__init__.py**

      📄 *File Path*: `./src/vectordb/__init__.py`

      ```
      
      ```

    - 📁 **chunkers/**
      - 📄 **SentenceChunker.py**

        📄 *File Path*: `./src/vectordb/chunkers/SentenceChunker.py`

        ```
        """
Sentence chunker. Based on Weaviate's Verba.
https://github.com/weaviate/Verba
"""
import spacy
from tqdm import tqdm
from loguru import logger
from typing_extensions import List

from src.vectordb.readers.document import Document
from src.vectordb.chunkers.chunk import Chunk
from src.vectordb.chunkers.interface import Chunker


class SentenceChunker(Chunker):
    """
    SentenceChunker for Verba built with spaCy.
    """

    def __init__(
        self,
        name="WordChunker",
        requires_library=["spacy"],
        requires_env=None,
        description="Chunk documents by sentences. You can specify how many sentences should overlap between chunks to improve retrieval.",
    ):
        """
        Initializes the SentenceChunker object.

        This method initializes the SentenceChunker object by calling the __init__ method of the parent class using the super() function. It sets the `name` attribute to "WordChunker", `requires_library` to ["spacy"], `default_units` to 3, `default_overlap` to 2, and `description` to "Chunk documents by sentences. You can specify how many sentences should overlap between chunks to improve retrieval." It then tries to set the `nlp` attribute to a blank spacy model for English with the "sentencizer" pipeline added, and if Spacy is not installed, it sets `nlp` to None.

        Parameters:
            None

        Returns:
            None
        """
        super().__init__(
            name=name,
            requires_library=requires_library,
            requires_env=requires_env,
            description=description,
        )
        self.default_units = 3
        self.default_overlap = 2
        try:
            self.nlp = spacy.blank("en")
            self.nlp.add_pipe("sentencizer")
        except ValueError as e:
            logger.warning(f"Spacy is not installed. Setting nlp to none. {e}")
            self.nlp = None

    def chunk(
        self, documents: List[Document], units: int, overlap: int
    ) -> List[Document]:
        """
        Chunk the given List of documents into smaller chunks based on the specified units and overlap.

        Args:
            documents (List[Document]): A List of Document objects representing the documents to be chunked.
            units (int): The number of tokens per chunk.
            overlap (int): The number of tokens that should overlap between chunks.

        Returns:
            List[Document]: The List of Document objects with their chunks added.
        """
        for document in tqdm(
            documents, total=len(documents), desc="Chunking documents"
        ):
            # Skip if document already contains chunks
            if len(document.chunks) > 0:
                continue
            if not self.nlp:
                continue
            if not document.text:
                continue
            doc = list(self.nlp(document.text).sents)

            if units > len(doc) or units < 1:
                logger.warning(
                    f"Unit value either exceeds length of actual document or is below 1 ({units}/{len(doc)})"
                )
                continue

            if overlap >= units:
                logger.warning(
                    f"Overlap value is greater than unit (Units {units}/ Overlap {overlap})"
                )
                continue

            i = 0
            split_id_counter = 0
            while i < len(doc):
                # Overlap
                start_i = i
                end_i = i + units
                end_i = min(end_i, len(doc))
                text = "".join(sent.text for sent in doc[start_i:end_i])
                doc_chunk = Chunk(
                    text=text,
                    doc_name=document.name or "document",
                    doc_type=document.doc_type or "document",
                    chunk_id=str(split_id_counter) or str(0),
                )
                document.chunks.append(doc_chunk)
                split_id_counter += 1

                # Exit loop if this was the last possible chunk
                if end_i == len(doc):
                    break

                i += units - overlap  # Step forward, considering overlap

        return documents

        ```

      - 📄 **TiktokenChunker.py**

        📄 *File Path*: `./src/vectordb/chunkers/TiktokenChunker.py`

        ```
        """
Tiktoken Token Chunker. Based on Weaviate's Verba.
https://github.com/weaviate/Verba
"""
import tiktoken
from tqdm import tqdm
from loguru import logger
from typing_extensions import List

from src.vectordb.chunkers.interface import Chunker
from src.vectordb.chunkers.chunk import Document, Chunk


class TokenChunker(Chunker):
    """
    TokenChunker built with tiktoken.
    """

    def __init__(self, name, requires_library, requires_env, description):
        """
        Initializes the TokenChunker class.

        This method initializes the TokenChunker class by calling the __init__ method of the parent class using the super() function. It sets the 'name' attribute to "TokenChunker", the 'requires_library' attribute to ["tiktoken"], the 'default_units' attribute to 250, and the 'default_overlap' attribute to 50. It also sets the 'description' attribute to "Chunk documents by tokens powered by tiktoken. You can specify how many tokens should overlap between chunks to improve retrieval." Finally, it sets the 'encoding' attribute to the encoding for the "gpt-3.5-turbo" model using the tiktoken library.

        Parameters:
            None

        Returns:
            None
        """
        super().__init__(
            name=name,
            requires_library=requires_library,
            requires_env=requires_env,
            description=description,
        )
        self.name = "TokenChunker"
        self.requires_library = ["tiktoken"]
        self.default_units = 250
        self.default_overlap = 50
        self.description = "Chunk documents by tokens powered by tiktoken. You can specify how many tokens should overlap between chunks to improve retrieval."
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def chunk(
        self, documents: List[Document], units: int, overlap: int
    ) -> List[Document]:
        """
        Chunk the given List of documents into smaller chunks based on the specified units and overlap.

        Args:
            documents (List[Document]): A List of Document objects representing the documents to be chunked.
            units (int): The number of tokens per chunk.
            overlap (int): The number of tokens that should overlap between chunks.

        Returns:
            List[Document]: The List of Document objects with their chunks added.

        Raises:
            None

        Description:
            This function takes a List of Document objects and chunks them into smaller chunks based on the specified units and overlap.
            Each Document object is checked to see if it already has chunks. If it does, the function skips it.
            If the Document does not have chunks, the function encodes the text of the document using the tiktoken encoding.
            If the number of tokens in the encoded text is less than or equal to the specified units, a single chunk is created with the entire text of the document.
            If the overlap is greater than or equal to the units, a warning is logged and the function continues to the next document.
            The function then iterates over the encoded tokens, creating chunks of the specified units and overlapping them by the specified overlap.
            Each chunk is created by decoding the corresponding tokens using the tiktoken encoding and creating a Chunk object with the decoded text, document name, document type, and a unique chunk ID.
            The function then appends the Chunk object to the Document object's List of chunks.
            The function returns the List of Document objects with their chunks added.
        """
        for document in tqdm(
            documents, total=len(documents), desc="Chunking documents"
        ):
            # Skip if document already contains chunks
            if len(document.chunks) > 0:
                continue

            encoded_tokens = self.encoding.encode(document.text, disallowed_special=())

            if units > len(encoded_tokens) or units < 1:
                doc_chunk = Chunk(
                    text=document.text,
                    doc_name=document.name,
                    doc_type=document.type,
                    chunk_id=0,
                )

            if overlap >= units:
                logger.warning(
                    f"Overlap value is greater than unit (Units {units}/ Overlap {overlap})"
                )
                continue

            i = 0
            split_id_counter = 0
            while i < len(encoded_tokens):
                # Overlap
                start_i = i
                end_i = min(i + units, len(encoded_tokens))

                chunk_tokens = encoded_tokens[start_i:end_i]
                chunk_text = self.encoding.decode(chunk_tokens)

                doc_chunk = Chunk(
                    text=chunk_text,
                    doc_name=document.name,
                    doc_type=document.type,
                    chunk_id=str(split_id_counter),
                )
                document.chunks.append(doc_chunk)
                split_id_counter += 1

                # Exit loop if this was the last possible chunk
                if end_i == len(encoded_tokens):
                    break

                i += units - overlap  # Step forward, considering overlap

        return documents

        ```

      - 📄 **WordChunker.py**

        📄 *File Path*: `./src/vectordb/chunkers/WordChunker.py`

        ```
        """
WordChunker for vectorstore. Based on Weaviate's Verba.
https://github.com/weaviate/Verba
"""
import spacy
from tqdm import tqdm
from loguru import logger
from typing_extensions import List

from src.vectordb.readers.document import Document
from src.vectordb.chunkers.chunk import Chunk
from src.vectordb.chunkers.interface import Chunker


class WordChunker(Chunker):
    """
    WordChunker for Verba built with spaCy.
    """

    def __init__(self, name, requires_library, requires_env, description):
        """
        Initializes the WordChunker object.

        This method initializes the WordChunker object by calling the __init__ method of the parent class using the super() function. It sets the `name` attribute to "WordChunker", `requires_library` to ["spacy"], `default_units` to 100, `default_overlap` to 50, `description` to "Chunk documents by words. You can specify how many words should overlap between chunks to improve retrieval." It then tries to set the `nlp` attribute to a blank spacy model for English, and if Spacy is not installed, it sets `nlp` to None.

        Parameters:
            None

        Returns:
            None
        """
        super().__init__(
            name=name,
            requires_library=requires_library,
            requires_env=requires_env,
            description=description,
        )
        self.name = "WordChunker"
        self.requires_library = ["spacy"]
        self.default_units = 100
        self.default_overlap = 50
        self.description = "Chunk documents by words. You can specify how many words should overlap between chunks to improve retrieval."
        try:
            self.nlp = spacy.blank("en")
        except ValueError as e:
            logger.warning(f"Spacy is not installed. Setting nlp to none. {e}")
        self.nlp = None

    def chunk(
        self, documents: List[Document], units: int, overlap: int
    ) -> List[Document]:
        """
        Chunk the given List of documents into smaller chunks based on the specified units and overlap.

        Args:
            documents (List[Document]): A List of Document objects representing the documents to be chunked.
            units (int): The number of words per chunk.
            overlap (int): The number of words that should overlap between chunks.

        Returns:
            List[Document]: The List of Document objects with their chunks added.

        Description:
            This function takes a List of Document objects and chunks them into smaller chunks based on the specified units and overlap.
            Each Document object is checked to see if it already has chunks. If it does, the function skips it.
            If the Document does not have chunks, the function uses the `nlp` attribute (a spacy model) to tokenize the text of the document.
            If the number of tokens in the document is less than or equal to the specified units, a single chunk is created with the entire text of the document.
            If the overlap is greater than or equal to the units, a warning is logged and the function continues to the next document.
            The function then iterates over the tokens, creating chunks of the specified units and overlapping them by the specified overlap.
            Each chunk is created by creating a Chunk object with the corresponding text, document name, document type, and a unique chunk ID.
            The function then appends the Chunk object to the Document object's List of chunks.
            The function returns the List of Document objects with their chunks added.
        """
        for document in tqdm(
            documents, total=len(documents), desc="Chunking documents"
        ):
            # Skip if document already contains chunks
            if len(document.chunks) > 0:
                continue

            if not document:
                raise ValueError("Document is empty")

            if self.nlp:
                doc = self.nlp(document.text or " document")

            if units > len(doc) or units < 1:
                doc_chunk = Chunk(
                    text=doc.text,
                    doc_name=document.name or "document",
                    doc_type=document.doc_type or "document",
                    chunk_id=str(0),
                )

            if overlap >= units:
                logger.warning(
                    f"Overlap value is greater than unit (Units {units}/ Overlap {overlap})"
                )
                continue

            i = 0
            split_id_counter = 0
            while i < len(doc):
                # Overlap
                start_i = i
                end_i = i + units
                end_i = min(end_i, len(doc))
                doc_chunk = Chunk(
                    text=doc[start_i:end_i].text,
                    doc_name=document.name or "document",
                    doc_type=document.doc_type or "document",
                    chunk_id=str(split_id_counter) or str(0),
                )
                document.chunks.append(doc_chunk)
                split_id_counter += 1

                # Exit loop if this was the last possible chunk
                if end_i == len(doc):
                    break

                i += units - overlap  # Step forward, considering overlap

        return documents

        ```

      - 📄 **__init__.py**

        📄 *File Path*: `./src/vectordb/chunkers/__init__.py`

        ```
        
        ```

      - 📄 **chunk.py**

        📄 *File Path*: `./src/vectordb/chunkers/chunk.py`

        ```
        """
Chunk. Based off of Weaviate's Verba.
https://github.com/weaviate/Verba
"""
from typing_extensions import Dict


class Chunk:
    "Chunk class that represents a chunk of text."

    def __init__(
        self,
        text: str = "",
        doc_name: str = "",
        doc_type: str = "",
        doc_uuid: str = "",
        chunk_id: str = "",
    ):
        """
        Initializes a new instance of the Chunk class.

        Args:
            text (str): The text of the chunk. Defaults to an empty string.
            doc_name (str): The name of the document containing the chunk. Defaults to an empty string.
            doc_type (str): The type of the document containing the chunk. Defaults to an empty string.
            doc_uuid (str): The UUID of the document containing the chunk. Defaults to an empty string.
            chunk_id (str): The ID of the chunk. Defaults to an empty string.
        """
        self._text = text
        self._doc_name = doc_name
        self._doc_type = doc_type
        self._doc_uuid = doc_uuid
        self._chunk_id = chunk_id
        self._text_no_overlap = text
        self._tokens = 0
        self._vector = None
        self._score = 0

    @property
    def text(self):
        """
        Returns the value of the `_text` attribute.

        :return: The value of the `_text` attribute.
        :rtype: str
        """
        return self._text

    @property
    def text_no_overlap(self):
        """
        Returns the value of the `_text_no_overlap` attribute.

        :return: The value of the `_text_no_overlap` attribute.
        :rtype: str
        """
        return self._text_no_overlap

    @property
    def doc_name(self):
        """
        Get the name of the document.

        :return: The name of the document.
        :rtype: str
        """
        return self._doc_name

    @property
    def doc_type(self):
        """
        Get the doc_type property of the object.

        Returns:
            str: The doc_type property of the object.
        """
        return self._doc_type

    @property
    def doc_uuid(self):
        """
        Returns the value of the `_doc_uuid` attribute.

        :return: The value of the `_doc_uuid` attribute.
        :rtype: str
        """
        return self._doc_uuid

    @property
    def chunk_id(self):
        """
        Get the chunk ID of the object.

        :return: The chunk ID of the object.
        :rtype: str
        """
        return self._chunk_id

    @property
    def tokens(self):
        """
        Returns the value of the `_tokens` attribute.
        """
        return self._tokens

    @property
    def vector(self):
        """
        Returns the value of the `_vector` attribute.

        :return: The value of the `_vector` attribute.
        """
        return self._vector

    @property
    def score(self):
        """
        Get the score property of the object.

        Returns:
            The score property of the object.
        """
        return self._score

    def set_uuid(self, uuid):
        """
        Sets the UUID of the document.

        Parameters:
            uuid (str): The UUID to set.

        Returns:
            None
        """
        self._doc_uuid = uuid

    def set_tokens(self, token):
        """
        Sets the value of the `_tokens` attribute.

        Parameters:
            token (any): The token to set.

        Returns:
            None
        """
        self._tokens = token

    def set_vector(self, vector):
        """
        Set the vector attribute of the object.

        Args:
            vector: The vector to set.

        Returns:
            None
        """
        self._vector = vector

    def set_score(self, score):
        """
        Set the score of the object.

        Parameters:
            score (any): The score to set.

        Returns:
            None
        """
        self._score = score

    def to_dict(self) -> Dict:
        """
        Convert the Chunk object to a Dictionary.
        """
        return {
            "text": self.text,
            "doc_name": self.doc_name,
            "doc_type": self.doc_type,
            "doc_uuid": self.doc_uuid,
            "chunk_id": self.chunk_id,
            "tokens": self.tokens,
            "vector": self.vector,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: Dict):
        """Construct a Chunk object from a Dictionary."""
        chunk = cls(
            text=data.get("text", ""),
            doc_name=data.get("doc_name", ""),
            doc_type=data.get("doc_type", ""),
            doc_uuid=data.get("doc_uuid", ""),
            chunk_id=data.get("chunk_id", ""),
        )
        chunk.set_tokens(data.get("tokens", 0))
        chunk.set_vector(data.get("vector"))
        chunk.set_score(data.get("score", 0))
        return chunk

        ```

      - 📄 **interface.py**

        📄 *File Path*: `./src/vectordb/chunkers/interface.py`

        ```
        from typing_extensions import List
from src.vectordb.readers.document import Document
from src.vectordb.readers.interface import InputForm
from src.vectordb.component import Component


class Chunker(Component):
    """
    Interface for Verba Chunking.
    """

    def __init__(self, name, description, requires_library, requires_env):
        """
        Initializes the Chunker object.

        This method initializes the Chunker object by calling the __init__ method of the parent class using the super() function. It sets the `input_form` attribute to the value of `InputForm.CHUNKER.value`, which represents the default input form for all chunkers. It also sets the `default_units` attribute to 100 and the `default_overlap` attribute to 50.

        Parameters:
            None

        Returns:
            None
        """
        super().__init__(
            name=name,
            requires_env=requires_env,
            requires_library=requires_library,
            description=description,
        )
        self.input_form = InputForm.CHUNKER.value  # Default for all Chunkers
        self.default_units = 100
        self.default_overlap = 50

    def chunk(
        self, documents: List[Document], units: int, overlap: int
    ) -> List[Document]:
        """
        Chunk verba documents into chunks based on units and overlap.

        This method takes a List of Verba documents, the number of units per chunk, and the overlap between chunks. It raises a NotImplementedError because the chunk method must be implemented by a subclass.

        Parameters:
            documents (List[Document]): A List of Verba documents to be chunked.
            units (int): The number of units per chunk (words, sentences, etc.).
            overlap (int): The amount of overlap between chunks.

        Returns:
            List[Document]: A List of chunked documents.

        Raises:
            NotImplementedError: If the chunk method is not implemented by a subclass.
        """
        raise NotImplementedError("chunk method must be implemented by a subclass.")

        ```

      - 📄 **manager.py**

        📄 *File Path*: `./src/vectordb/chunkers/manager.py`

        ```
        """
Chunker Manager. Manager class that handles chunking classes. Based one Weaviate's Verba.
https://github.com/weaviate/Verba
"""
import tiktoken
from loguru import logger
from typing_extensions import List, Dict

from src.vectordb.readers.document import Document
from src.vectordb.chunkers.interface import Chunker
from src.vectordb.chunkers.SentenceChunker import SentenceChunker
from src.vectordb.chunkers.TiktokenChunker import TokenChunker
from src.vectordb.chunkers.WordChunker import WordChunker


class ChunkerManager:
    """
    Chunker Manager class. Handles chunking classes.
    """

    def __init__(self):
        """
        Initializes a new instance of the ChunkerManager class.

        This method initializes the instance variables of the ChunkerManager class. It creates an empty List called `batch`, sets the `token_count` variable to 0, and creates an empty List called `batches`. It also creates a Dictionary called `chunker` with three key-value pairs, where the keys are strings representing the names of different chunking classes ("TokenChunker", "WordChunker", and "SentenceChunker"), and the values are instances of the corresponding chunking classes (`TokenChunker`, `WordChunker`, and `SentenceChunker`). The `selected_chunker` variable is then set to the instance of the `TokenChunker` class.

        Parameters:
            None

        Returns:
            None
        """
        self.batch = []
        self.token_count = 0
        self.batches = []

        self.chunker: Dict[str, Chunker] = {
            "TokenChunker": TokenChunker(),
            "WordChunker": WordChunker(),
            "SentenceChunker": SentenceChunker(),
        }
        self.selected_chunker: Chunker = self.chunker["TokenChunker"]

    def chunk(
        self, documents: List[Document], units: int, overlap: int
    ) -> List[Document]:
        """
        Chunk verba documents into chunks based on units and overlap.

        This method takes a List of Verba documents, the number of units per chunk, and the overlap between chunks. It uses the selected chunker to chunk the documents and returns a List of the chunked documents. If the chunked documents pass the check for the token count, the List of chunked documents is returned. Otherwise, an empty List is returned.

        Parameters:
            documents (List[Document]): A List of Verba documents to be chunked.
            units (int): The number of units per chunk (words, sentences, etc.).
            overlap (int): The amount of overlap between chunks.

        Returns:
            List[Document]: A List of chunked documents if the chunked documents pass the check for the token count. Otherwise, an empty List.
        """
        chunked_docs = self.selected_chunker.chunk(documents, units, overlap)
        logger.info("Chunking completed")
        return chunked_docs if self.check_chunks(chunked_docs) else []

    def set_chunker(self, chunker: str) -> bool:
        """
        Set the selected chunker based on the given chunker name.

        Parameters:
            chunker (str): The name of the chunker to be set.

        Returns:
            bool: True if the chunker is found and set successfully, False otherwise.
        """
        if chunker in self.chunker:
            self.selected_chunker = self.chunker[chunker]
            return True
        else:
            logger.warning(f"Chunker {chunker} not found")
            return False

    def get_chunkers(self) -> Dict[str, Chunker]:
        """
        Returns a Dictionary containing all the chunkers available.

        :return: A Dictionary where the keys are the names of the chunkers and the values are the chunkers themselves.
        :return type: Dict[str, Chunker]
        """
        return self.chunker

    def check_chunks(self, documents: List[Document]) -> int:
        """
        Checks the token count of chunks in a List of Verba documents.

        This function takes a List of Verba documents as input and checks the token count of each chunk in the documents. It uses the selected chunker to encode the text of each chunk and calculates the token count. The function hardcaps the token count of each chunk to 1000 tokens.

        Parameters:
            documents (List[Document]): A List of Verba documents to be checked.

        Returns:
            int: The number of batches created from the chunks in the documents. If no batches are created, 0 is returned.

        Raises:
            IndexError: If no batches are created.
        """

        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

        try:
            for document in documents:
                chunks = document.chunks
                for chunk in chunks:
                    tokens = encoding.encode(chunk.text, disallowed_special=())
                    chunk.set_tokens(tokens)
                    while self.token_count < 1000:
                        for token in tokens:
                            self.token_count += 1
                            self.batch.append(token)
                            if len(self.batch) >= 1000:
                                self.batch = []
                                self.token_count = 0
                                break
                            self.batches.append(self.batch)
        except IndexError as e:
            logger.error(f"no batchs created.{e}")
            return 0
        return len(self.batches)

        ```

    - 📄 **component.py**

      📄 *File Path*: `./src/vectordb/component.py`

      ```
      from typing_extensions import List
from pydantic import BaseModel


class Component(BaseModel):
    name: str
    requires_env: List[str]
    requires_library: List[str]
    description: str

      ```

    - 📁 **embedders/**
      - 📄 **ADAEmbedder.py**

        📄 *File Path*: `./src/vectordb/embedders/ADAEmbedder.py`

        ```
        """
ADAEmbedder. Based on Weaviate's Verba.
"""
import os
import openai
from weaviate.client import Client
from typing_extensions import List
from dotenv import load_dotenv

from src.vectordb.embedders.interface import Embedder
from src.vectordb.readers.document import Document

load_dotenv()


class ADAEmbedder(Embedder):
    """
    ADAEmbedder for Verba.
    """

    def __init__(self) -> None:
        """
        Constructor for ADAEmbedder class. Initializes the name, required environment variables,
        required libraries, and description for the embedder. Sets the vectorizer to 'text2vec-openai'.
        """
        super().__init__(
            name="ADAEmbedder",
            requires_env=["OPENAI_API_KEY"],
            requires_library=["openai"],
            description="Embeds and retrieves objects using OpenAI's ADA model",
        )
        self.vectorizer = "text2vec-openai"
        self.openai = openai.OpenAI()
        self.openai.api_key = str(os.getenv("OPENAI_API_KEY"))

    def embed(
        self, documents: List[Document], client: Client, batch_size: int = 100
    ) -> bool:
        """
        Embeds the given list of documents and their chunks into Weaviate using the SentenceTransformer model.
        Parameters:
            documents (List[Document]): A list of Document objects representing the documents to be embedded.
            client (Client): The Weaviate client used to import the embedded data.
        Returns:
            bool
        """
        return self.import_data(documents, client)

    def vectorize_query(self, query: str):
        return self.openai.embeddings.create(
            input=[query], model="text-embedding-ada-002"
        )

        ```

      - 📄 **CohereEmbedder.py**

        📄 *File Path*: `./src/vectordb/embedders/CohereEmbedder.py`

        ```
        """
Cohere Embedder. Based on Weaviate's Verba.
https://github.com/weaviate/Verba
"""
from weaviate.client import Client
from typing_extensions import List

from src.vectordb.embedders.interface import Embedder
from src.vectordb.readers.document import Document


class CohereEmbedder(Embedder):
    """
    CohereEmbedder for Verba.
    """

    def __init__(self):
        """
        Initializes a new instance of the class.

        This method initializes the CohereEmbedder class with the necessary parameters for embedding and retrieving
        objects using Cohere's ember multilingual-v2.0 model. It sets the description, name, requires_env, and
        requires_library attributes. The requires_env attribute is set to a list containing the "COHERE_API_KEY"
        environment variable, while the requires_library attribute is set to None. The vectorizer attribute is set
        to "text2vec-cohere".

        Parameters:
            None

        Returns:
            None
        """
        super().__init__(
            description=(
                "Embeds and retrieves objects using Cohere's ember multilingual-v2.0 model"
            ),
            name="CohereEmbedder",
            requires_env=["COHERE_API_KEY"],
            requires_library=None,
        )
        self.vectorizer = "text2vec-cohere"
        self.tokenizer = CohereEmbedder()

    def embed(
        self, documents: List[Document], client: Client, batch_size: int = 100
    ) -> bool:
        """
        Embeds the given list of documents and their chunks into Weaviate using the SentenceTransformer model.

        Parameters:
            documents (List[Document]): A list of Document objects representing the documents to be embedded.
            client (Client): The Weaviate client used to import the embedded data.
            batch_size (int, optional): The batch size for embedding the documents. Defaults to 100.

        Returns:
            bool: True if the embedding and import were successful, False otherwise.
        """
        return self.tokenizer.embed(documents, client)

        ```

      - 📄 **MiniLMEmbedder.py**

        📄 *File Path*: `./src/vectordb/embedders/MiniLMEmbedder.py`

        ```
        """
Mini LM Embedder. Based on Weaviate's Verba.
https://github.com/weaviate/Verba
"""
from tqdm import tqdm
from weaviate import Client
import torch
from transformers import AutoModel, AutoTokenizer
from typing import List, Union
from loguru import logger

from goldenverba.components.embedding.interface import Embedder
from goldenverba.components.reader.document import Document


class MiniLMEmbedder(Embedder):
    """
    MiniLMEmbedder for Verba.
    """

    model: AutoModel
    tokenizer: AutoTokenizer

    def __init__(self):
        """
        Initializes the MiniLMEmbedder class.

        This function initializes the MiniLMEmbedder class by setting the name, required libraries, description, and vectorizer attributes. It also attempts to get the device on which the model will be run. If a CUDA-enabled GPU is available, it uses that device. If not, it checks if the Multi-Process Service (MPS) is available and uses that device. If neither a CUDA device nor an MPS device is available, it falls back to using the CPU.

        The function then loads the pre-trained model and tokenizer from the "sentence-transformers/all-MiniLM-L6-v2" repository using the AutoModel and AutoTokenizer classes from the transformers library. The model and tokenizer are moved to the device obtained earlier.

        If there is a RuntimeError during the initialization process, a warning message is logged.
        Parameters:
            None

        Returns:
            None
        """
        super().__init__()
        self.name = "MiniLMEmbedder"
        self.requires_library = ["torch", "transformers"]
        self.description = "Embeds and retrieves objects using SentenceTransformer's all-MiniLM-L6-v2 model"
        self.vectorizer = "MiniLM"

        try:

            def get_device():
                """
                Returns the appropriate device for running the model based on the availability of CUDA-enabled GPUs and Multi-Process Service (MPS).

                :return: A torch.device object representing the device to be used for running the model.
                :rtype: torch.device
                """
                if torch.cuda.is_available():
                    return torch.device("cuda")
                elif torch.backends.mps.is_available():
                    return torch.device("mps")
                else:
                    return torch.device("cpu")

            self.device = get_device()

            self.model = AutoModel.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2", device_map=self.device
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
                device_map=self.device,
            )
            self.model = self.model.to(self.device)

        except RuntimeError as e:
            logger.warning(str(e))

    def embed(
        self,
        documents: List[Document],
        client: Client,
    ) -> bool:
        """
        Embeds the given list of documents and their chunks into Weaviate using the SentenceTransformer model.

        Parameters:
            documents (List[Document]): A list of Document objects representing the documents to be embedded.
            client (Client): The Weaviate client used to import the embedded data.

        Returns:
            bool: True if the embedding and import were successful, False otherwise.
        """
        for document in tqdm(
            documents, total=len(documents), desc="Vectorizing document chunks"
        ):
            for chunk in document.chunks:
                chunk.set_vector(self.vectorize_chunk(chunk.text))

        return self.import_data(documents, client)

    def vectorize_chunk(self, chunk) -> Union[List[float], None]:
        """
        Vectorize a chunk of text into a list of floats representing the average embedding of the tokens in the chunk.

        Parameters:
            chunk (str): The text chunk to be vectorized.

        Returns:
            List[float]: A list of floats representing the average embedding of the tokens in the chunk.

        Raises:
            RuntimeError: If there is an error creating the embeddings.
        """
        try:
            text = chunk
            tokens = self.tokenizer.tokenize(text)

            max_length = (
                self.tokenizer.model_max_length
            )  # Get the max sequence length for the model
            batches = []
            batch = []
            token_count = 0

            for token in tokens:
                token_length = len(
                    self.tokenizer.encode(token, add_special_tokens=False)
                )
                if token_count + token_length <= max_length:
                    batch.append(token)
                    token_count += token_length
                else:
                    batches.append(" ".join(batch))
                    batch = [token]
                    token_count = token_length

            # Don't forget to add the last batch
            if batch:
                batches.append(" ".join(batch))

            embeddings = []

            for batch in batches:
                inputs = self.tokenizer(
                    text=batch, return_tensors="pt", padding=True, truncation=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                # Taking the mean of the hidden states to obtain an embedding for the batch
                embedding = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(embedding)

            # Concatenate the embeddings to make averaging easier
            all_embeddings = torch.cat(embeddings)

            averaged_embedding = all_embeddings.mean(dim=0)

            return averaged_embedding.tolist()
        except RuntimeError as e:
            logger.warning(str(e))

    def vectorize_query(self, query: str) -> List[float]:
        """
        Vectorizes a query by calling the vectorize_chunk method and returns the resulting vector.

        :param query: The query to be vectorized.
        :type query: str
        :return: A list of floats representing the vectorized query.
        :rtype: List[float]
        """
        return self.vectorize_chunk(query)

        ```

      - 📄 **SentenceEmbedder.py**

        📄 *File Path*: `./src/vectordb/embedders/SentenceEmbedder.py`

        ```
        """
Mini LM Embedder. Based on Weaviate's Verba.
https://github.com/weaviate/Verba
"""
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from typing import List
from weaviate.client import Client
from loguru import logger

from src.vectordb.readers.document import Document
from src.vectordb.embedders.interface import Embedder


class MiniLMEmbedder(Embedder):
    """
    MiniLMEmbedder for Verba.
    """

    model = AutoModel
    tokenizer = AutoTokenizer

    def __init__(self):
        """
        Initializes the MiniLMEmbedder class.

        This function initializes the MiniLMEmbedder class by setting the name, required libraries, description, and vectorizer attributes. It also attempts to get the device on which the model will be run. If a CUDA-enabled GPU is available, it uses that device. If not, it checks if the Multi-Process Service (MPS) is available and uses that device. If neither a CUDA device nor an MPS device is available, it falls back to using the CPU.

        The function then loads the pre-trained model and tokenizer from the "sentence-transformers/all-MiniLM-L6-v2" repository using the AutoModel and AutoTokenizer classes from the transformers library. The model and tokenizer are moved to the device obtained earlier.

        If there is a RuntimeError during the initialization process, a warning message is logged.

        Parameters:
            None

        Returns:
            None
        """
        super().__init__()
        self.name = "MiniLMEmbedder"
        self.requires_library = ["torch", "transformers"]
        self.description = "Embeds and retrieves objects using SentenceTransformer's all-MiniLM-L6-v2 model"
        self.vectorizer = "MiniLM"
        try:

            def get_device():
                """
                Returns the appropriate device for running the model based on the availability of CUDA-enabled GPUs and Multi-Process Service (MPS).

                :return: A torch.device object representing the device to be used for running the model.
                :rtype: torch.device
                """
                if torch.cuda.is_available():
                    return torch.device("cuda")
                elif torch.backends.mps.is_available():
                    return torch.device("mps")
                else:
                    return torch.device("cpu")

            self.device = get_device()

            self.model = AutoModel.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2", device_map=self.device
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2", device_map=self.device
            )
            self.model = self.model.to(self.device)

        except RuntimeError as e:
            logger.warning(str(e))

    def embed(
        self,
        documents: List[Document],
        client: Client,
    ) -> bool:
        """
        Embeds the given list of documents and their chunks into Weaviate using the SentenceTransformer model.

        Parameters:
            documents (List[Document]): A list of Document objects representing the documents to be embedded.
            client (Client): The Weaviate client used to import the embedded data.

        Returns:
            bool: True if the embedding and import were successful, False otherwise.
        """
        for document in tqdm(
            documents, total=len(documents), desc="Vectorizing document chunks"
        ):
            for chunk in document.chunks:
                chunk.set_vector(self.vectorize_chunk(chunk.text))

        return self.import_data(documents, client)

    def vectorize_chunk(self, chunk) -> List[float]:
        """
        Vectorize a chunk of text into a list of floats representing the average embedding of the tokens in the chunk.

        Parameters:
            chunk (str): The text chunk to be vectorized.

        Returns:
            List[float]: A list of floats representing the average embedding of the tokens in the chunk.

        Raises:
            RuntimeError: If there is an error creating the embeddings.
        """
        try:
            text = chunk
            tokens = self.tokenizer.tokenize(text)
            max_length = (
                self.tokenizer.model_max_length
            )  # Get the max sequence length for the model
            batches = []
            batch = []
            token_count = 0

            for token in tokens:
                token_length = len(
                    self.tokenizer.encode(token, add_special_tokens=False)
                )
                if token_count + token_length <= max_length:
                    batch.append(token)
                else:
                    batches.append(" ".join(batch))
                    batch = [token]
                    token_count = token_length

            # Don't forget to add the last batch
            if batch:
                batches.append(" ".join(batch))

            embeddings = []

            for batch in batches:
                inputs = self.tokenizer(
                    text=batch, return_tensors="pt", padding=True, truncation=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                # Taking the mean of the hidden states to obtain an embedding for the batch
                embedding = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(embedding)

            # Concatenate the embeddings to make averaging easier
            all_embeddings = torch.cat(embeddings)

            averaged_embedding = all_embeddings.mean(dim=0)

            return averaged_embedding.tolist()
        except Exception as e:
            raise RuntimeError(f"Error creating embeddings: {e}") from e

    def vectorize_query(self, query: str) -> List[float]:
        """
        Vectorize a query by calling the vectorize_chunk method and return the resulting vector.

        Parameters:
            query (str): The query to be vectorized.

        Returns:
            List[float]: A list of floats representing the vectorized query.
        """
        return self.vectorize_chunk(query)

        ```

      - 📄 **__init__.py**

        📄 *File Path*: `./src/vectordb/embedders/__init__.py`

        ```
        
        ```

      - 📄 **interface.py**

        📄 *File Path*: `./src/vectordb/embedders/interface.py`

        ```
        """
Embedder Interface. Based on Weaviate's Verba.
https://github.com/weaviate/Verba
"""
from tqdm import tqdm
from typing import List, Dict, Tuple, Union, Optional
from weaviate import Client
from loguru import logger
from src.vectordb.readers.document import Document
from src.vectordb.readers.interface import InputForm
from src.vectordb.component import Component
from src.vectordb.schema.schema_generator import (
    VECTORIZERS,
    EMBEDDINGS,
    strip_non_letters,
)


class Embedder(Component):
    """
    Interface for Verba Embedding.
    """

    def __init__(
        self,
        name: str,
        description: str,
        requires_env: List[str],
        requires_library: Optional[List[str]] = None,
    ):
        """
        Initializes a new instance of the class.

        Args:
            name (str): The name of the instance.
            description (str): The description of the instance.
            requires_env (List[str]): A list of environment variables required by the instance.
            requires_library (List[str]): A list of libraries required by the instance.

        Returns:
            None
        """
        super().__init__(
            name=name,
            description=description,
            requires_env=requires_env,
            requires_library=requires_library,
        )
        self.input_form = InputForm.TEXT.value  # Default for all Embedders
        self.vectorizer = ""

    def embed(
        self, documents: List[Document], client: Client, batch_size: int = 100
    ) -> bool:
        """
        A method for embedding data using the given client and batch size.

        Args:
            self: The object instance.
            client (Client): The Weaviate client used to interact with the Weaviate server.
            batch_size (int): The size of the batch for embedding data. Default is 100.

        Returns:
            bool: Indicates whether the embedding was successful.

        Raises:
            NotImplementedError: If the embed method is not implemented by a subclass.
        """
        raise NotImplementedError("embed method must be implemented by a subclass.")

    def import_data(
        self,
        documents: List[Document],
        client: Client,
    ) -> bool:
        """
        Imports data into the Weaviate client.

        Args:
            self (Embedder): The Embedder instance.
            documents (List[Document]): The list of Document objects to import.
            client (Client): The Weaviate client.

        Returns:
            bool: True if the data is successfully imported, False otherwise.

        Raises:
            ValueError: If an exception occurs during the data import process.
        """
        try:
            if self.vectorizer not in VECTORIZERS and self.vectorizer not in EMBEDDINGS:
                logger.warning(f"Vectorizer of {self.name} not found")
                return False

            for i, document in enumerate(documents):
                batches = []
                uuid = ""
                temp_batch = []
                token_counter = 0
                for chunk in document.chunks:
                    if not chunk or not chunk.tokens:
                        break
                    if token_counter + chunk.tokens <= 5000:
                        token_counter += chunk.tokens
                        temp_batch.append(chunk)
                    else:
                        batches.append(temp_batch.copy())
                        token_counter = chunk.tokens
                        temp_batch = [chunk]
                if len(temp_batch) > 0:
                    batches.append(temp_batch.copy())
                    token_counter = 0
                    temp_batch = []

                logger.info(
                    f"({i+1}/{len(documents)}) Importing document {document.name} with {len(batches)} batches"
                )

                with client.batch as batch:
                    batch.batch_size = 1
                    properties = {
                        "text": str(document.text),
                        "doc_name": str(document.name),
                        "doc_type": str(document.type),
                        "doc_link": str(document.link),
                        "chunk_count": len(document.chunks),
                        "timestamp": str(document.timestamp),
                    }

                    class_name = f"Document_{strip_non_letters(self.vectorizer)}"
                    uuid = client.batch.add_data_object(properties, class_name)

                    for chunk in document.chunks:
                        chunk.set_uuid(uuid)

                chunk_count = 0
                for _batch_id, chunk_batch in tqdm(
                    enumerate(batches), total=len(batches), desc="Importing batches"
                ):
                    with client.batch as batch:
                        batch.batch_size = len(chunk_batch)
                        for chunk in chunk_batch:
                            chunk_count += 1

                            properties = {
                                "text": chunk.text,
                                "doc_name": str(document.name),
                                "doc_uuid": chunk.doc_uuid,
                                "doc_type": chunk.doc_type,
                                "chunk_id": chunk.chunk_id,
                            }
                            class_name = f"Chunk_{strip_non_letters(self.vectorizer)}"

                            # Check if vector already exists
                            if chunk.vector is None:
                                client.batch.add_data_object(properties, class_name)
                            else:
                                client.batch.add_data_object(
                                    properties, class_name, vector=chunk.vector
                                )
                if not document.name:
                    document.__setattr__(name="name", value="uuid", obj=document)
                self.check_document_status(
                    client,
                    uuid,
                    str(document.name),
                    f"Document_{strip_non_letters(self.vectorizer)}",
                    f"Chunk_{strip_non_letters(self.vectorizer)}",
                    len(document.chunks),
                )
            return True
        except Exception as e:
            raise ValueError(e) from e

    def check_document_status(
        self,
        client: Client,
        doc_uuid: str,
        doc_name: str,
        doc_class_name: str,
        chunk_class_name: str,
        chunk_count: int,
    ):
        """
        Verifies that imported documents and its chunks exist in the database, if not, remove everything that was added and rollback.

        :param client: Client - Weaviate Client
        :param doc_uuid: str - Document UUID
        :param doc_name: str - Document name
        :param doc_class_name: str - Class name of Document
        :param chunk_class_name: str - Class name of Chunks
        :param chunk_count: int - Number of expected chunks
        :raises ValueError: If the document is not found or the chunk mismatch occurs
        :return: None
        """
        document = client.data_object.get_by_id(
            doc_uuid,
            class_name=doc_class_name,
        )

        if document is None:
            raise ValueError(f"Document {doc_uuid} not found {document}")
        results = (
            client.query.get(
                class_name=chunk_class_name,
                properties=[
                    "doc_name",
                ],
            )
            .with_where(
                {
                    "path": ["doc_uuid"],
                    "operator": "Equal",
                    "valueText": doc_uuid,
                }
            )
            .with_limit(chunk_count + 1)
            .do()
        )

        if len(results["data"]["Get"][chunk_class_name]) != chunk_count:
            # Rollback if fails
            self.remove_document(client, doc_name, doc_class_name, chunk_class_name)
            raise ValueError(
                f"Chunk mismatch for {doc_uuid} {len(results['data']['Get'][chunk_class_name])} != {chunk_count}"
            )

    def remove_document(
        self, client: Client, doc_name: str, doc_class_name: str, chunk_class_name: str
    ) -> None:
        """
        Removes a document and its chunks from the database.

        Args:
            client (Client): The Weaviate client used to interact with the database.
            doc_name (str): The name of the document to be removed.
            doc_class_name (str): The class name of the document.
            chunk_class_name (str): The class name of the chunks.

        Returns:
            None
        """
        client.batch.delete_objects(
            class_name=doc_class_name,
            where={"path": ["doc_name"], "operator": "Equal", "valueText": doc_name},
        )

        client.batch.delete_objects(
            class_name=chunk_class_name,
            where={"path": ["doc_name"], "operator": "Equal", "valueText": doc_name},
        )

        logger.warning(f"Deleted document {doc_name} and its chunks")

    def remove_document_by_id(self, client: Client, doc_id: str):
        """
        Removes a document and its chunks from the database by their document ID.

        Args:
            client (Client): The Weaviate client used to interact with the database.
            doc_id (str): The ID of the document to be removed.

        Returns:
            None
        """
        doc_class_name = f"Document_{strip_non_letters(self.vectorizer)}"
        chunk_class_name = f"Chunk_{strip_non_letters(self.vectorizer)}"

        client.data_object.delete(uuid=doc_id, class_name=doc_class_name)

        client.batch.delete_objects(
            class_name=chunk_class_name,
            where={"path": ["doc_uuid"], "operator": "Equal", "valueText": doc_id},
        )

        logger.warning(f"Deleted document {doc_id} and its chunks")

    def get_document_class(self) -> str:
        """
        Returns the document class name based on the vectorizer.

        :return: A string representing the document class name.
        :rtype: str
        """
        return f"Document_{strip_non_letters(self.vectorizer)}"

    def get_chunk_class(self) -> str:
        """
        Returns the chunk class name based on the vectorizer.

        :return: A string representing the chunk class name.
        :rtype: str
        """
        return f"Chunk_{strip_non_letters(self.vectorizer)}"

    def get_cache_class(self) -> str:
        """
        Returns the cache class name based on the vectorizer.

        :return: A string representing the cache class name.
        :rtype: str
        """
        return f"Cache_{strip_non_letters(self.vectorizer)}"

    def search_documents(self, client: Client, query: str, doc_type: str) -> List:
        """
        Search for documents from Weaviate based on the given query and document type.

        Args:
            client (Client): The Weaviate client used to query the database.
            query (str): The search query.
            doc_type (str): The document type to search for. If None, all document types will be searched.

        Returns:
            List: A list of documents matching the search query.
        """
        doc_class_name = f"Document_{strip_non_letters(self.vectorizer)}"

        if not doc_type or doc_type is None:
            query_results = (
                client.query.get(
                    class_name=doc_class_name,
                    properties=["doc_name", "doc_type", "doc_link"],
                )
                .with_bm25(query, properties=["doc_name"])
                .with_additional(properties=["id"])
                .with_limit(100)
                .do()
            )
        else:
            query_results = (
                client.query.get(
                    class_name=doc_class_name,
                    properties=["doc_name", "doc_type", "doc_link"],
                )
                .with_bm25(query, properties=["doc_name"])
                .with_where(
                    {
                        "path": ["doc_type"],
                        "operator": "Equal",
                        "valueText": doc_type,
                    }
                )
                .with_additional(properties=["id"])
                .with_limit(100)
                .do()
            )

        return query_results["data"]["Get"][doc_class_name]

    def get_need_vectorization(self) -> bool:
        """
        Check if the current vectorizer is in the list of embeddings.

        Returns:
            bool: True if the vectorizer is in the list of embeddings, False otherwise.
        """
        return self.vectorizer in EMBEDDINGS

    def vectorize_query(self, query: str):
        """
        Vectorizes a query by calling the vectorize_chunk method and returns the resulting vector.

        :param query: The query to be vectorized.
        :type query: str
        :return: A list of floats representing the vectorized query.
        :rtype: List[float]
        :raises NotImplementedError: If the vectorize_query method is not implemented by a subclass.
        """
        raise NotImplementedError(
            "vectorize_query method must be implemented by a subclass."
        )

    def conversation_to_query(self, queries: List[str], conversation: Dict) -> str:
        """
        Converts a conversation to a query string by extracting relevant information from the conversation and joining it with the provided queries.

        Parameters:
            queries (List[str]): A list of queries.
            conversation (Dict): A dictionary representing a conversation, where each key-value pair represents a message in the conversation.

        Returns:
            str: The resulting query string.

        Example:
            >>> queries = ["What is the weather like?", "How cold is it?"]
            >>> conversation = {
            ...     0: {"type": "user", "content": "What is the weather like today?"},
            ...     1: {"type": "system", "content": "The weather is cold and rainy."},
            ...     2: {"type": "user", "content": "How cold is it?"}
            ... }
            >>> conversation_to_query(queries, conversation)
            'the weather like today ? how cold is it ?'
        """
        query = ""

        if len(conversation) > 1:
            if conversation[-1].type == "system":
                query += f"{conversation[-1].content} "
            elif conversation[-2].type == "system":
                query += f"{conversation[-2].content} "

        for _query in queries:
            query += f"{_query} "

        return query.lower()

    def retrieve_semantic_cache(
        self, client: Client, query: str, dist: float = 0.04
    ) -> Union[str, Tuple[Union[str, None], Union[float, None]]]:
        """
        Retrieve results from semantic cache based on query and distance threshold.

        :param client: The client object used to query the semantic cache.
        :type client: Client
        :param query: The query string to search for in the semantic cache.
        :type query: str
        :param dist: The distance threshold for the semantic cache search. Defaults to 0.04.
        :type dist: float
        :return: A tuple containing the system response and the distance between the query and the cached query, if found. Otherwise, returns (None, None).
        :rtype: Union[Tuple[str, float], Tuple[None, None]]
        """
        needs_vectorization = self.get_need_vectorization()

        match_results = (
            client.query.get(
                class_name=self.get_cache_class(),
                properties=["query", "system"],
            )
            .with_where(
                {
                    "path": ["query"],
                    "operator": "Equal",
                    "valueText": query,
                }
            )
            .with_limit(1)
        ).do()
        if not match_results["data"]:
            return None, None
        if (
            "data" in match_results
            and len(match_results["data"]["Get"][self.get_cache_class()]) > 0
            and (
                query
                == match_results["data"]["Get"][self.get_cache_class()][0]["query"]
            )
        ):
            logger.info("Direct match from cache")
            if not match_results["data"]["Get"][self.get_cache_class()][0]["system"]:
                return None, None
            return (
                match_results["data"]["Get"][self.get_cache_class()][0]["system"],
                0.0,
            )

        query_results = (
            client.query.get(
                class_name=self.get_cache_class(),
                properties=["query", "system"],
            )
            .with_additional(properties=["distance"])
            .with_limit(1)
        )

        if needs_vectorization:
            vector = self.vectorize_query(query)
            query_results = query_results.with_near_vector(
                content={"vector": vector},
            ).do()

        else:
            query_results = query_results.with_near_text(
                content={"concepts": [query]},
            ).do()

        if "data" not in query_results:
            logger.warning(query_results)
            return None, None

        results = query_results["data"]["Get"][self.get_cache_class()]

        if not results:
            return None, None

        result = results[0]

        if float(result["_additional"]["distance"]) > dist:
            return None, None
        logger.info("Retrieved similar from cache")
        return result["system"], float(result["_additional"]["distance"])

    def add_to_semantic_cache(self, client: Client, query: str, system: str):
        """
        Adds a query and its corresponding system response to the semantic cache.

        Parameters:
            client (Client): The Weaviate client used to interact with the semantic cache.
            query (str): The query string to be added to the semantic cache.
            system (str): The system response corresponding to the query.

        Returns:
            None
        """
        needs_vectorization = self.get_need_vectorization()

        with client.batch as batch:
            batch.batch_size = 1
            properties = {"query": query, "system": system}
            logger.info("Saved to cache")

            if needs_vectorization:
                vector = self.vectorize_query(query)
                client.batch.add_data_object(
                    properties, self.get_cache_class(), vector=vector
                )
            else:
                client.batch.add_data_object(properties, self.get_cache_class())

        ```

      - 📄 **manager.py**

        📄 *File Path*: `./src/vectordb/embedders/manager.py`

        ```
        """
Embedding Manager for handling the Embedder classes. Based on Weaviate's Verba.
https://github.com/weaviate/Verba
"""
from weaviate.client import Client
from typing import List
from loguru import logger

from typing_extensions import Dict
from src.vectordb.embedders.interface import Embedder
from src.vectordb.readers.document import Document
from src.vectordb.embedders.ADAEmbedder import ADAEmbedder
from src.vectordb.embedders.CohereEmbedder import CohereEmbedder
from src.vectordb.embedders.SentenceEmbedder import SentenceEmbedder
from src.vectordb.embedders.MiniLMEmebdder import MiniLMEmbedder


class EmbeddingManager:
    """
    Embedding Manager for handling embedder classes.
    """

    def __init__(self):
        """
        Constructor for EmbeddingManager class. Initializes the embedders dictionary with different Embedder instances and sets the selected embedder to ADAEmbedder by default.
        """
        self.embedders: Dict[str, Embedder] = {
            "MiniLMEmbedder": MiniLMEmbedder(),
            "ADAEmbedder": ADAEmbedder(),
            "CohereEmbedder": CohereEmbedder(),
            "SentenceEmbedder": SentenceEmbedder(),
        }
        self.selected_embedder: Embedder = self.embedders["ADAEmbedder"]

    def embed(
        self, documents: List[Document], client: Client, batch_size: int = 100
    ) -> bool:
        """
        Embeds a list of Verba documents and its chunks to Weaviate.

        Args:
            documents (List[Document]): List of Verba documents to be embedded.
            client (Client): Weaviate client used for embedding.
            batch_size (int, optional): Batch size of the input. Defaults to 100.

        Returns:
            bool: True if the embedding was successful, False otherwise.
        """
        return self.selected_embedder.embed(documents, client, batch_size)

    def set_embedder(self, embedder: str) -> bool:
        """
        Sets the selected embedder for embedding documents and chunks to Weaviate.

        Args:
            embedder (str): The name of the embedder to set.

        Returns:
            bool: True if the embedder is found and set successfully, False otherwise.
        """
        if embedder in self.embedders:
            self.selected_embedder = self.embedders[embedder]
            return True
        else:
            logger.warning(f"Embedder {embedder} not found")
            return False

    def get_embedders(self) -> Dict[str, Embedder]:
        """
        Get the dictionary of embedders.

        Returns:
            Dict[str, Embedder]: A dictionary where the keys are strings representing the names of the embedders and the values are instances of the Embedder class.
        """
        return self.embedders

        ```

    - 📁 **embeddings/**
      - 📄 **manager.py**

        📄 *File Path*: `./src/vectordb/embeddings/manager.py`

        ```
        from src.vectordb.embeddings.tik_tokenizer import TikTokenizer
from src.vectordb.weaviate_manager import Embedder
from src.

class EmbeddingManager:
    def __init__(self):
        self.embedders: dict[str, Embedder] = {
            "MiniLMEmbedder": TikTokenizer(),
        }
        self.selected_embedder: Embedder = self.embedders["ADAEmbedder"]

    def embed(
        self, documents: list[Document], client: Client, batch_size: int = 100
    ) -> bool:
        """Embed verba documents and its chunks to Weaviate
        @parameter: documents : list[Document] - List of Verba documents
        @parameter: client : Client - Weaviate Client
        @parameter: batch_size : int - Batch Size of Input
        @returns bool - Bool whether the embedding what successful.
        """
        return self.selected_embedder.embed(documents, client)

    def set_embedder(self, embedder: str) -> bool:
        if embedder in self.embedders:
            self.selected_embedder = self.embedders[embedder]
            return True
        else:
            msg.warn(f"Embedder {embedder} not found")
            return False

    def get_embedders(self) -> dict[str, Embedder]:
        return self.embedders

        ```

      - 📄 **tik_tokenizer.py**

        📄 *File Path*: `./src/vectordb/embeddings/tik_tokenizer.py`

        ```
        from pydantic import BaseModel
from loguru import logger
from typing import List, Optional
from src.data_models.generators import TokenizerConfig


input_config = TokenizerConfig()


class TikTokenizer(Tokenizer):
    def __init__(self, input_config: Optional[Tokenizer] = input_config):
        super().__init__(input_config or self.config)

    def install_dependencies(self):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as e:
            logger.warn(str(e))

    def setup_tokenizer(self):
        try:
            self.device = self.get_device()

            self.model = AutoModel.from_pretrained(
                self.model_path, device_map=self.device
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, device_map=self.device
            )
            self.model = self.model.to(self.device)
        except Exception as e:
            logger.warn(str(e))

    def get_device():
        try:
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        except Exception as e:
            logger.warn(str(e))

    def tokenize(self, chunk) -> List[float]:
        try:
            return self.tokenizer.tokenize(chunk)
        except Exception as e:
            logger.warn(str(e))

    def embed(
        self,
        tokens=None,
    ):
        max_length = self.tokenizer.model_max_length
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        if len(tokens) < max_length:
            tokens = tokens + ["[PAD]"] * (max_length - len(tokens))

        encoded_input = self.tokenizer(
            tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        encoded_input = encoded_input.to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        model_output = model_output.last_hidden_state.squeeze(0)
        return model_output.tolist()

    def vectorize_chunk(self, chunk) -> List[float]:
        try:
            import torch

            tokens = self.tokenize(chunk)
            return self.embed(tokens)
        except Exception as e:
            logger.warn(str(e))

        ```

      - 📁 **todo/**
        - 📄 **ada_embedder.py**

          📄 *File Path*: `./src/vectordb/embeddings/todo/ada_embedder.py`

          ```
          class ADAEmbedder(Embedder):
    """
    ADAEmbedder for Verba.
    """

    def __init__(self):
        super().__init__()
        self.name = "ADAEmbedder"
        self.requires_env = ["OPENAI_API_KEY"]
        self.requires_library = ["openai"]
        self.description = "Embeds and retrieves objects using OpenAI's ADA model"
        self.vectorizer = "text2vec-openai"

    def embed(
        self,
        documents: list[Document],
        client: Client,
    ) -> bool:
        """Embed verba documents and its chunks to Weaviate
        @parameter: documents : list[Document] - List of Verba documents
        @parameter: client : Client - Weaviate Client
        @parameter: batch_size : int - Batch Size of Input
        @returns bool - Bool whether the embedding what successful.
        """
        return self.import_data(documents, client)

          ```

        - 📄 **cohere_embedder.py**

          📄 *File Path*: `./src/vectordb/embeddings/todo/cohere_embedder.py`

          ```
          class CohereEmbedder(Embedder):
    """
    CohereEmbedder for Verba.
    """

    def __init__(self):
        super().__init__()
        self.name = "CohereEmbedder"
        self.requires_env = ["COHERE_API_KEY"]
        self.description = (
            "Embeds and retrieves objects using Cohere's embed-multilingual-v2.0 model"
        )
        self.vectorizer = "text2vec-cohere"

    def embed(
        self,
        documents: list[Document],
        client: Client,
    ) -> bool:
        """Embed verba documents and its chunks to Weaviate
        @parameter: documents : list[Document] - List of Verba documents
        @parameter: client : Client - Weaviate Client
        @parameter: batch_size : int - Batch Size of Input
        @returns bool - Bool whether the embedding what successful.
        """
        return self.import_data(documents, client)

          ```

        - 📄 **minilm_embedder.py**

          📄 *File Path*: `./src/vectordb/embeddings/todo/minilm_embedder.py`

          ```
          class MiniLMEmbedder(Embedder):
    """
    MiniLMEmbedder for Verba.
    """

    def __init__(self):
        super().__init__()
        self.name = "MiniLMEmbedder"
        self.requires_library = ["torch", "transformers"]
        self.description = "Embeds and retrieves objects using SentenceTransformer's all-MiniLM-L6-v2 model"
        self.vectorizer = "MiniLM"
        self.model = None
        self.tokenizer = None
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            def get_device():
                if torch.cuda.is_available():
                    return torch.device("cuda")
                elif torch.backends.mps.is_available():
                    return torch.device("mps")
                else:
                    return torch.device("cpu")

            self.device = get_device()

            self.model = AutoModel.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2", device_map=self.device
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2", device_map=self.device
            )
            self.model = self.model.to(self.device)

        except Exception as e:
            msg.warn(str(e))

    def embed(
        self,
        documents: list[Document],
        client: Client,
    ) -> bool:
        """Embed verba documents and its chunks to Weaviate
        @parameter: documents : list[Document] - List of Verba documents
        @parameter: client : Client - Weaviate Client
        @parameter: batch_size : int - Batch Size of Input
        @returns bool - Bool whether the embedding what successful.
        """
        for document in tqdm(
            documents, total=len(documents), desc="Vectorizing document chunks"
        ):
            for chunk in document.chunks:
                chunk.set_vector(self.vectorize_chunk(chunk.text))

        return self.import_data(documents, client)

    def vectorize_chunk(self, chunk) -> list[float]:
        try:
            import torch

            text = chunk
            tokens = self.tokenizer.tokenize(text)

            max_length = (
                self.tokenizer.model_max_length
            )  # Get the max sequence length for the model
            batches = []
            batch = []
            token_count = 0

            for token in tokens:
                token_length = len(
                    self.tokenizer.encode(token, add_special_tokens=False)
                )
                if token_count + token_length <= max_length:
                    batch.append(token)
                    token_count += token_length
                else:
                    batches.append(" ".join(batch))
                    batch = [token]
                    token_count = token_length

            # Don't forget to add the last batch
            if batch:
                batches.append(" ".join(batch))

            embeddings = []

            for batch in batches:
                inputs = self.tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                # Taking the mean of the hidden states to obtain an embedding for the batch
                embedding = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(embedding)

            # Concatenate the embeddings to make averaging easier
            all_embeddings = torch.cat(embeddings)

            averaged_embedding = all_embeddings.mean(dim=0)

            return averaged_embedding.tolist()
        except Exception:
            raise

    def vectorize_query(self, query: str) -> list[float]:
        return self.vectorize_chunk(query)

          ```

    - 📁 **readers/**
      - 📄 **UnstructuredReader.py**

        📄 *File Path*: `./src/vectordb/readers/UnstructuredReader.py`

        ```
        """
UnstructuredPDF Reader. Based on Weaviate's Verba.
https://github.com/weaviate/Verba
"""
import base64
import glob
import os
from datetime import datetime
from pathlib import Path
from loguru import logger
import requests
from wasabi import msg
from typing_extensions import List, Optional

from src.vectordb.readers.document import Document
from src.vectordb.readers.interface import InputForm, Reader


class UnstructuredPDF(Reader):
    """
    UnstructuredPDF Reader that handles ORC of PDF files. Does require an API key from https://unstructured.io/api-key-hosted. Should be included in your .env file as UNSTRUCTURED_API_KEY.
    """

    def __init__(self):
        """
        Initializes an instance of the UnstructuredPDF class.

        This method initializes the attributes of the UnstructuredPDF class, including the file types,
        required environment variables, name, description, and input form.

        Parameters:
            None

        Returns:
            None
        """
        super().__init__()
        self.file_types = [".pdf"]
        self.requires_env = ["UNSTRUCTURED_API_KEY"]
        self.name = "UnstructuredPDF"
        self.description = "Reads PDF files powered by unstructured.io"
        self.input_form = InputForm.UPLOAD.value

    def load(
        self,
        bites: Optional[List[str]] = None,
        contents: Optional[List[str]] = None,
        paths: Optional[List[str]] = None,
        file_names: Optional[List[str]] = None,
        document_type: str = "Documentation",
    ) -> List[Document]:
        """
        Load the content from the provided paths, bites, or contents and return a List of Document objects.

        Args:
            bites (List[str], optional): A List of byte strings representing the content to be loaded. Defaults to None.
            contents (List[str], optional): A List of strings representing the content to be loaded. Defaults to None.
            paths (List[str], optional): A List of file paths representing the files to be loaded. Defaults to None.
            file_names (List[str], optional): A List of strings representing the names of the files to be loaded. Defaults to None.
            document_type (str, optional): The type of the document. Defaults to "Documentation".

        Returns:
            List[Document]: A List of Document objects representing the loaded content.

        Raises:
            None

        Examples:
            >>> reader = UnstructuredPDF()
            >>> documents = reader.load(paths=["/path/to/file.pdf"])
            >>> documents = reader.load(bites=["bites_string"], file_names=["file.pdf"])
            >>> documents = reader.load(contents=["content_string"], file_names=["file.txt"])
        """
        if file_names is None:
            file_names = []
        if paths is None:
            paths = []
        if contents is None:
            contents = []
        if bites is None:
            bites = []
        documents = []

        # If paths exist
        if paths:
            for path in paths:
                if path != "":
                    data_path = Path(path)
                    if data_path.exists():
                        if data_path.is_file():
                            documents += self.load_file(data_path, document_type)
                        else:
                            documents += self.load_directory(data_path, document_type)
                    else:
                        msg.warn(f"Path {data_path} does not exist")

        # If bites exist
        if bites and len(bites) == len(file_names):
            for byte, file_name in zip(bites, file_names):
                documents += self.load_bites(byte, file_name, document_type)

        # If content exist
        if contents and len(contents) == len(file_names):
            for content, file_name in zip(contents, file_names):
                document = Document(
                    name=file_name,
                    text=content,
                    doc_type=document_type,
                    timestamp=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    reader=self.name,
                )
                documents.append(document)

        logger.info(f"Loaded {len(documents)} documents")
        return documents

    def load_bites(self, bites_string, file_name, document_type: str) -> List[Document]:
        """
        Loads the content from the provided `bites_string` by decoding it, making a POST request to the `UNSTRUCTURED_API_URL` with the decoded content, and processing the response to create a Document object. The created Document is returned in a List.

        Parameters:
            bites_string: A string containing the content to be loaded.
            file_name: The name of the file.
            document_type: The type of the document.

        Returns:
            A List containing a single Document object.
        """
        url = os.environ.get(
            "UNSTRUCTURED_API_URL", "https://api.unstructured.io/general/v0/general"
        )

        headers = {
            "accept": "application/json",
            "unstructured-api-key": os.environ.get("UNSTRUCTURED_API_KEY", ""),
        }

        data = {
            "strategy": "auto",
        }

        decoded_bites = base64.b64decode(bites_string)
        with open("reconstructed.pdf", "wb") as file:
            file.write(decoded_bites)

        file_data = {"files": open("reconstructed.pdf", "rb")}

        response = requests.post(
            url, headers=headers, data=data, files=file_data, timeout=30
        )

        json_response = response.json()

        full_content = ""

        for chunk in json_response:
            if "text" in chunk:
                text = chunk["text"]
                full_content += f"{text} "

        document = Document(
            text=full_content,
            doc_type=document_type,
            name=str(file_name),
            link=str(file_name),
            timestamp=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            reader=self.name,
        )
        documents = [document]
        logger.info(f"Loaded {str(file_name)}")
        os.remove("reconstructed.pdf")
        return documents

    def load_file(self, file_path: Path, document_type: str) -> List[Document]:
        """
        Load a file and return a List of Document objects.

        Args:
            file_path (Path): The path to the file to be loaded.
            document_type (str): The type of the document.

        Returns:
            List[Document]: A List of Document objects representing the loaded file.
        """
        if file_path.suffix not in self.file_types:
            logger.warning(f"{file_path.suffix} not supported")
            return []

        url = os.environ.get(
            "UNSTRUCTURED_API_URL", "https://api.unstructured.io/general/v0/general"
        )

        headers = {
            "accept": "application/json",
            "unstructured-api-key": os.environ.get("UNSTRUCTURED_API_KEY", ""),
        }

        data = {
            "strategy": "auto",
        }

        file_data = {"files": open(file_path, "rb")}

        response = requests.post(
            url, headers=headers, data=data, files=file_data, timeout=30
        )

        file_data["files"].close()

        json_response = response.json()

        full_content = ""

        for chunk in json_response:
            if "text" in chunk:
                text = chunk["text"]
                full_content += f"{text} "

        document = Document(
            text=full_content,
            doc_type=document_type,
            name=str(file_path),
            link=str(file_path),
            timestamp=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            reader=self.name,
        )
        documents = [document]
        logger.info(f"Loaded {str(file_path)}")
        return documents

    def load_directory(self, dir_path: Path, document_type: str) -> List[Document]:
        """
        Initialize an empty Dictionary to store the file contents
        Parameters:
            self: The UnstructuredPDF instance
            dir_path (Path): The path to the directory to load files from
            document_type (str): The type of the document

        Returns:
            List[Document]: A list of Document objects representing the loaded files
        """
        # Initialize an empty Dictionary to store the file contents
        documents = []

        # Convert dir_path to string, in case it's a Path object
        dir_path_str = str(dir_path)

        # Loop through each file type
        for file_type in self.file_types:
            # Use glob to find all the files in dir_path and its subdirectories matching the current file_type
            files = glob.glob(f"{dir_path_str}/**/*{file_type}", recursive=True)

            # Loop through each file
            for file in files:
                logger.info(f"Reading {str(file)}")
                with open(file, encoding="utf-8"):
                    documents += self.load_file(Path(file), document_type=document_type)

        logger.info(f"Loaded {len(documents)} documents")
        return documents

        ```

      - 📄 **__init__.py**

        📄 *File Path*: `./src/vectordb/readers/__init__.py`

        ```
        
        ```

      - 📄 **document.py**

        📄 *File Path*: `./src/vectordb/readers/document.py`

        ```
        """
Document class for Vectordb. Based on Weaviate's Verba.
https://github.com/weaviate/Verba
"""
from typing_extensions import List, Dict, Optional

from src.vectordb.chunkers.chunk import Chunk


class Document:
    """
    Document class. Standard document for data ingestion into Vectordb.
    """

    def __init__(
        self,
        text: Optional[str] = "",
        doc_type: Optional[str] = "",
        name: Optional[str] = "",
        path: Optional[str] = "",
        link: Optional[str] = "",
        timestamp: Optional[str] = "",
        reader: Optional[str] = "",
        meta: Optional[Dict] = None,
    ):
        """
        Initializes a new instance of the Document class.

        Args:
            text (str): The text of the document. Defaults to an empty string.
            doc_type (str): The doc_type of the document. Defaults to an empty string.
            name (str): The name of the document. Defaults to an empty string.
            path (str): The path of the document. Defaults to an empty string.
            link (str): The link of the document. Defaults to an empty string.
            timestamp (str): The timestamp of the document. Defaults to an empty string.
            reader (str): The reader of the document. Defaults to an empty string.
            meta (Dict): Additional metadata of the document. Defaults to an empty dictionary.

        Returns:
            None
        """
        if meta is None:
            meta = {}
        self._text = text
        self._doc_type = doc_type
        self._name = name
        self._path = path
        self._link = link
        self._timestamp = timestamp
        self._reader = reader
        self._meta = meta
        self.chunks: List[Chunk] = []

    @property
    def text(self):
        """
        Get the text property of the object.

        Returns:
            str: The text property of the object.
        """
        return self._text

    @property
    def doc_type(self):
        """
        Get the doc_type property of the object.

        Returns:
            str: The doc_type property of the object.
        """
        return self._doc_type

    @property
    def name(self):
        """
        Get the name property of the object.

        Returns:
            str: The name property of the object.
        """
        return self._name

    @property
    def path(self):
        """
        Get the path property of the object.

        Returns:
            str: The path property of the object.
        """
        return self._path

    @property
    def link(self):
        """
        Get the link property of the object.

        Returns:
            str: The link property of the object.
        """
        return self._link

    @property
    def timestamp(self):
        """
        Get the timestamp property of the object.

        Returns:
            str: The timestamp property of the object.
        """
        return self._timestamp

    @property
    def reader(self):
        """
        Get the reader property of the object.

        Returns:
            The reader property of the object.
        """
        return self._reader

    @property
    def meta(self):
        """
        Get the meta property of the object.

        Returns:
            The meta property of the object.
        """
        return self._meta

    @staticmethod
    def to_json(document) -> Dict:
        """
        Convert the Document object to a JSON dictionary.

        Args:
            document: The Document object to be converted.

        Returns:
            A JSON dictionary representing the Document object.
        """
        return {
            "text": document.text,
            "doc_type": document.doc_type,
            "name": document.name,
            "path": document.path,
            "link": document.link,
            "timestamp": document.timestamp,
            "reader": document.reader,
            "meta": document.meta,
            "chunks": [chunk.to_Dict() for chunk in document.chunks],
        }

    @staticmethod
    def from_json(doc_Dict: Dict):
        """
        Convert a JSON string to a Document object.

        Args:
            doc_Dict (Dict): A dictionary containing the JSON string.

        Returns:
            Document: A Document object created from the JSON string.
        """
        document = Document(
            text=doc_Dict.get("text", ""),
            doc_type=doc_Dict.get("doc_type", ""),
            name=doc_Dict.get("name", ""),
            path=doc_Dict.get("path", ""),
            link=doc_Dict.get("link", ""),
            timestamp=doc_Dict.get("timestamp", ""),
            reader=doc_Dict.get("reader", ""),
            meta=doc_Dict.get("meta", {}),
        )
        # Assuming Chunk has a from_dict method
        document.chunks = [
            Chunk.from_dict(chunk_data) for chunk_data in doc_Dict.get("chunks", [])
        ]
        return document

        ```

      - 📄 **githubreader.py**

        📄 *File Path*: `./src/vectordb/readers/githubreader.py`

        ```
        """
Github Reader. Based on Weaviate's Verba.
https://github.com/weaviate/Verba
"""
import base64
import json
import os
import requests
from datetime import datetime
from wasabi import msg
from typing_extensions import List, Tuple, Optional
from loguru import logger

from src.vectordb.readers.document import Document
from src.vectordb.readers.interface import InputForm, Reader


class GithubReader(Reader):
    """
    The GithubReader downloads files from Github and ingests them into Weaviate.
    """

    def __init__(self) -> None:
        """
        Initializes a new instance of the GithubReader class.

        This constructor sets up the initial state of the GithubReader object by assigning values to its instance variables. It calls the constructor of the parent class using the `super()` function. The `name` variable is set to "GithubReader", indicating the name of the reader. The `requires_env` variable is set to a List containing the required environment variable "GITHUB_TOKEN". The `description` variable is set to a string describing the functionality of the GithubReader, indicating that it downloads only text files from a GitHub repository and ingests them into Verba, and provides the format for specifying the repository, owner, and folder. The `input_form` variable is set to the value of `InputForm.INPUT.value`, indicating that the reader expects input in the form of a specific input.

        Parameters:
            None

        Returns:
            None
        """
        super().__init__()
        self.name = "GithubReader"
        self.requires_env = ["GITHUB_TOKEN"]
        self.description = "Downloads only text files from a GitHub repository and ingests it into Verba. Use this format {owner}/{repo}/{folder}"
        self.input_form = InputForm.INPUT.value

    def load(
        self,
        bites: Optional[List[str]] = None,
        contents: Optional[List[str]] = None,
        paths: Optional[List[str]] = None,
        file_names: Optional[List[str]] = None,
        document_type: str = "Documentation",
    ) -> List[Document]:
        """
        Load documents from the given paths, contents, bites, and file names.

        Parameters:
            bites (Optional[List[str]]): A list of base64-encoded strings representing data. Defaults to None.
            contents (Optional[List[str]]): A list of strings containing document contents. Defaults to None.
            paths (Optional[List[str]]): A list of file paths to load data from. Defaults to None.
            file_names (Optional[List[str]]): A list of file names corresponding to the paths. Defaults to None.
            document_type (str): The type of document to load. Defaults to "Documentation".

        Returns:
            List[Document]: A list of Document objects loaded from the data sources.
        """
        if file_names is None:
            file_names = []
        if paths is None:
            paths = []
        if contents is None:
            contents = []
        if bites is None:
            bites = []
        documents = []

        # If paths exist
        if len(paths) > 0:
            for path in paths:
                if path != "":
                    files = self.fetch_docs(path)

                    for _file in files:
                        try:
                            content, link, _path = self.download_file(path, _file)
                        except TypeError as e:
                            logger.warning(f"Couldn't load, skipping {_file}: {str(e)}")
                            continue

                        if ".json" in _file:
                            json_obj = json.loads(str(content))
                            try:
                                document = Document.from_json(json_obj)
                            except Exception as e:
                                raise ValueError(f"Loading JSON failed {e}") from e

                        else:
                            document = Document(
                                text=content,
                                type=document_type,
                                name=_file,
                                link=link,
                                path=_path,
                                timestamp=str(
                                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                ),
                                reader=self.name,
                            )
                        documents.append(document)

        msg.good(f"Loaded {len(documents)} documents")
        return documents

    def fetch_docs(self, path: str) -> List:
        """
        Fetches documents based on the given path from a GitHub repository.

        Parameters:
            path (str): The path to fetch documents from in the format owner/repo.

        Returns:
            List: A list of file paths for the fetched documents.
        """
        split = path.split("/")
        owner = split[0]
        repo = split[1]
        folder_path = "/".join(split[2:]) if len(split) > 2 else ""
        # Path should be owner/repo
        url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1"
        response = self.call_download_file(url)
        files = [
            item["path"]
            for item in response.json()["tree"]
            if item["path"].startswith(folder_path)
            and (
                item["path"].endswith(".md")
                or item["path"].endswith(".mdx")
                or item["path"].endswith(".txt")
                or item["path"].endswith(".json")
            )
        ]
        msg.info(
            f"Fetched {len(files)} file_names from {url} (checking folder {folder_path})"
        )
        return files

    def download_file(self, path: str, file_path: str) -> Tuple[str, str, str]:
        """
        Download files from Github based on file_name.

        Parameters:
            path (str): The path to a GitHub repository.
            file_path (str): The path of the file in the repository.

        Returns:
            str: The content of the file.

        Raises:
            None
        """
        split = path.split("/")
        owner = split[0]
        repo = split[1]

        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
        response = self.call_download_file(url)
        content_b64 = response.json()["content"]
        link = response.json()["html_url"]
        path = response.json()["path"]
        content = base64.b64decode(content_b64).decode("utf-8")
        logger.info(f"Downloaded {url}")
        return (content, link, path)

    def call_download_file(self, url) -> requests.Response:
        """
        Downloads a file from the given URL using the Github API.

        Args:
            url (str): The URL of the file to download.

        Returns:
            requests.Response: The response object containing the downloaded file.

        Raises:
            requests.exceptions.HTTPError: If the HTTP request returns an error status code.

        """
        headers = {
            "Authorization": f"token {os.environ.get('GITHUB_TOKEN', '')}",
            "Accept": "application/vnd.github.v3+json",
        }
        result = requests.get(url, headers=headers, timeout=30)
        result.raise_for_status()
        return result

        ```

      - 📄 **interface.py**

        📄 *File Path*: `./src/vectordb/readers/interface.py`

        ```
        """
Reader Interface. Based on Weaviate's Verba.
https://github.com/weaviate/Verba
"""
from enum import Enum
from typing_extensions import List

from src.vectordb.component import Component
from src.vectordb.readers.document import Document


class InputForm(Enum):
    UPLOAD = "UPLOAD"  # Input Form to upload text files directly
    INPUT = "INPUT"  # Simple Text Input in Frontend
    CHUNKER = "CHUNKER"  # Default Input for Chunkers
    TEXT = "TEXT"  # Default Input for Embedder


class Reader(Component):
    """
    Interface for Verba Readers.
    """

    def __init__(
        self,
        name,
        description,
        requires_library,
        requires_env,
    ):
        """
        Initializes a new instance of the class.

        Args:
            name (str): The name of the instance.
            description (str): The description of the instance.
            requires_library (List[str]): A list of libraries required by the instance.
            requires_env (List[str]): A list of environment variables required by the instance.

        Returns:
            None
        """
        super().__init__(
            name=name,
            description=description,
            requires_library=requires_library,
            requires_env=requires_env,
        )
        self.file_types = []
        self.input_form = InputForm.UPLOAD.value

    def load(
        self,
        bites: List[str],
        contents: List[str],
        paths: List[str],
        file_names: List[str],
        document_type: str,
    ) -> List[Document]:
        """
        Load the data from the given sources and return a list of Document objects.

        Args:
            bites (List[str]): A list of base64-encoded strings representing data.
            contents (List[str]): A list of strings containing document contents.
            paths (List[str]): A list of file paths to load data from.
            file_names (List[str]): A list of file names corresponding to the paths.
            document_type (str): The type of document to load.

        Returns:
            List[Document]: A list of Document objects loaded from the data sources.

        Raises:
            NotImplementedError: If the load method is not implemented by a subclass.
        """
        raise NotImplementedError("load method must be implemented by a subclass.")

        ```

      - 📄 **pdfreader.py**

        📄 *File Path*: `./src/vectordb/readers/pdfreader.py`

        ```
        """
PDF Reader that handles ORC of PDF files. Based on Weaviate's Verba
https://github.com/weaviate/Verba
"""
import base64
import glob
import os
from datetime import datetime
from pathlib import Path
from PyPDF2 import PdfReader
from wasabi import msg
from typing_extensions import List, Optional
from loguru import logger

from src.vectordb.readers.document import Document
from src.vectordb.readers.interface import InputForm, Reader


class PDFReader(Reader):
    """
    The PDFReader reads .pdf files using Unstructured. It can handle both paths, content and bites. Requires the PyPDF2 library.
    """

    def __init__(self):
        """
        Initializes the PDFReader object with the necessary file types, required libraries, name, description, and input form.
        """
        super().__init__()
        self.file_types = [".pdf"]
        self.requires_library = ["PyPDF2"]
        self.name = "PDFReader"
        self.description = "Reads PDF files using the PyPDF2 library"
        self.input_form = InputForm.UPLOAD.value

    def load(
        self,
        bites: Optional[List[str]] = None,
        contents: Optional[List[str]] = None,
        paths: Optional[List[str]] = None,
        file_names: Optional[List[str]] = None,
        document_type: str = "Documentation",
    ) -> List[Document]:
        """
        Ingest data into Weaviate
        @parameter: bites : List[str] - List of bites
        @parameter: contents : List[str] - List of string content
        @parameter: paths : List[str] - List of paths to files
        @parameter: file_names : List[str] - List of file names
        @parameter: document_type : str - Document type
        @returns List[Document] - Lists of documents.
        """
        if file_names is None:
            file_names = []
        if paths is None:
            paths = []
        if contents is None:
            contents = []
        if bites is None:
            bites = []
        documents = []

        # If paths exist
        if len(paths) > 0:
            for path in paths:
                if path != "":
                    data_path = Path(path)
                    if data_path.exists():
                        if data_path.is_file():
                            documents += self.load_file(data_path, document_type)
                        else:
                            documents += self.load_directory(data_path, document_type)
                    else:
                        msg.warn(f"Path {data_path} does not exist")

        # If bites exist
        if len(bites) > 0 and len(bites) == len(file_names):
            for byte, file_name in zip(bites, file_names):
                decoded_bites = base64.b64decode(byte)
                with open(f"{file_name}", "wb") as file:
                    file.write(decoded_bites)

                documents += self.load_file(Path(file_name), document_type)
                os.remove(f"{file_name}")

        # If content exist
        if len(contents) > 0 and len(contents) == len(file_names):
            for content, file_name in zip(contents, file_names):
                document = Document(
                    name=file_name,
                    text=content,
                    doc_type=document_type,
                    timestamp=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    reader=self.name,
                )
                documents.append(document)

        logger.info(f"Loaded {len(documents)} documents")
        return documents

    def load_file(self, file_path: Path, document_type: str) -> List[Document]:
        """
        Loads a PDF file and returns a list of Document objects.

        Args:
            file_path (Path): The path to the PDF file to be loaded.
            document_type (str): The type of document to load.

        Returns:
            List[Document]: A list of Document objects loaded from the PDF file.

        This function loads a PDF file based on the provided file path and returns a list of Document objects.
        It creates a PdfReader object from the file_path and extracts the full text from all the pages of the PDF.
        It then creates a Document object with the extracted text, document type, name, link, timestamp, and reader name.
        The created Document object is added to a list of documents.
        Finally, it logs a success message and returns the list of documents.
        """
        reader = PdfReader(file_path)

        full_text = "".join(page.extract_text() + "\n\n" for page in reader.pages)
        document = Document(
            text=full_text,
            doc_type=document_type,
            name=str(file_path),
            link=str(file_path),
            timestamp=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            reader=self.name,
        )
        documents = [document]
        logger.info(f"Loaded {str(file_path)}")
        return documents

    def load_directory(self, dir_path: Path, document_type: str) -> List[Document]:
        """
        Loads documents from a directory and its subdirectories.

        Args:
            dir_path (Path): The path to the directory.
            document_type (str): The type of the documents.

        Returns:
            List[Document]: A list of Document objects representing the loaded documents.

        This function initializes an empty list to store the documents. It then converts the `dir_path` to a string if it is a Path object.

        Next, it loops through each file type specified in `self.file_types`. For each file type, it uses the `glob` module to find all the files in `dir_path` and its subdirectories that match the current file type.

        For each file found, it logs a message indicating that it is reading the file. It then opens the file in read mode with UTF-8 encoding. It calls the `load_file` method with the file path and the specified `document_type`, and appends the returned documents to the `documents` list.

        Finally, it logs a success message indicating the number of documents loaded and returns the list of documents.
        """
        # Initialize an empty Dictionary to store the file contents
        documents = []

        # Convert dir_path to string, in case it's a Path object
        dir_path_str = str(dir_path)

        # Loop through each file type
        for file_type in self.file_types:
            # Use glob to find all the files in dir_path and its subdirectories matching the current file_type
            files = glob.glob(f"{dir_path_str}/**/*{file_type}", recursive=True)

            # Loop through each file
            for file in files:
                logger.info(f"Reading {str(file)}")
                with open(file, encoding="utf-8"):
                    documents += self.load_file(Path(file), document_type=document_type)

        logger.info(f"Loaded {len(documents)} documents")
        return documents

        ```

      - 📄 **simplereader.py**

        📄 *File Path*: `./src/vectordb/readers/simplereader.py`

        ```
        """
Simple Reader for .txt, .md, .mdx, and .json files. Based on Weaviate's Verba,
https://github.com/weaviate/Verba
"""
import base64
import glob
import json
from datetime import datetime
from pathlib import Path
from typing_extensions import List, Optional
from loguru import logger

from src.vectordb.readers.document import Document
from src.vectordb.readers.interface import InputForm, Reader


class SimpleReader(Reader):
    """
    The SimpleReader reads .txt, .md, .mdx, and .json files. It can handle both paths, content and bites.
    """

    def __init__(self):
        """
        Initializes a new instance of the SimpleReader class.

        This constructor sets up the initial state of the SimpleReader object by assigning values to its instance variables. It calls the constructor of the parent class using the `super()` function. The `file_types` variable is set to a List of file extensions that the SimpleReader can handle, which includes ".txt", ".md", ".mdx", and ".json". The `name` variable is set to "SimpleReader", indicating the name of the reader. The `description` variable is set to "Reads text, markdown, and json files.", providing a brief description of the reader's functionality. Finally, the `input_form` variable is set to the value of `InputForm.UPLOAD`, indicating that the reader expects input in the form of file uploads.

        Parameters:
            None

        Returns:
            None
        """
        super().__init__()
        self.file_types = [".txt", ".md", ".mdx", ".json"]
        self.name = "SimpleReader"
        self.description = "Reads text, markdown, and json files."
        self.input_form = InputForm.UPLOAD.value

    def load(
        self,
        bites: Optional[List[str]] = None,
        contents: Optional[List[str]] = None,
        paths: Optional[List[str]] = None,
        file_names: Optional[List[str]] = None,
        document_type: str = "Documentation",
    ) -> List[Document]:
        """
        Loads data from various sources and returns a list of Document objects.

        Parameters:
            bites (List[str]): A list of base64-encoded strings representing data.
            contents (List[str]): A list of strings containing document contents.
            paths (List[str]): A list of file paths to load data from.
            file_names (List[str]): A list of file names corresponding to the paths.
            document_type (str): The type of document to load. Defaults to "Documentation".

        Returns:
            List[Document]: A list of Document objects loaded from the data sources.
        """
        if file_names is None:
            file_names = []
        if paths is None:
            paths = []
        if contents is None:
            contents = []
        if bites is None:
            bites = []
        documents = []

        # If paths exist
        if paths:
            for path in paths:
                if path != "":
                    data_path = Path(path)
                    if data_path.exists():
                        if data_path.is_file():
                            documents += self.load_file(data_path, document_type)
                        else:
                            documents += self.load_directory(data_path, document_type)
                    else:
                        logger.warning(f"Path {data_path} does not exist")

        # If bytes exist
        if bites and len(bites) == len(file_names):
            for byte, file_name in zip(bites, file_names):
                decoded_bites = base64.b64decode(byte)
                try:
                    original_text = decoded_bites.decode("utf-8")
                except UnicodeDecodeError:
                    logger.info(
                        f"Error decoding text for file {file_name}. The file might not be a text file."
                    )
                    continue

                if ".json" in file_name:
                    json_obj = json.loads(original_text)
                    try:
                        document = Document.from_json(json_obj)
                    except Exception as e:
                        raise ValueError(f"Loading JSON failed {e}") from e

                else:
                    document = Document(
                        name=file_name,
                        text=original_text,
                        doc_type=document_type,
                        timestamp=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                        reader=self.name,
                    )
                documents.append(document)

        # If content exist
        if contents and len(contents) == len(file_names):
            for content, file_name in zip(contents, file_names):
                document = Document(
                    name=file_name,
                    text=content,
                    doc_type=document_type,
                    timestamp=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    reader=self.name,
                )
                documents.append(document)

        logger.info(f"Loaded {len(documents)} documents")
        return documents

    def load_file(self, file_path: Path, document_type: str) -> List[Document]:
        """
        Loads a file and returns a list of Document objects.

        Args:
            file_path (Path): The path to the file to be loaded.
            document_type (str): The type of document to load.

        Returns:
            List[Document]: A list of Document objects loaded from the file.

        Raises:
            Exception: If loading the JSON file fails.

        This function loads a file based on the provided file path and returns a list of Document objects.
        It first checks if the file extension is supported by the reader. If not, it logs a warning message and returns an empty list.
        If the file extension is supported, it opens the file in read mode with UTF-8 encoding.
        If the file is a JSON file, it loads the JSON object and creates a Document object from it.
        If the file is not a JSON file, it reads the entire file content and creates a Document object with the file content, document type, name, link, timestamp, and reader name.
        The created Document object is added to the list of documents.
        Finally, it logs a success message and returns the list of documents.
        """
        documents = []

        if file_path.suffix not in self.file_types:
            logger.warning(f"{file_path.suffix} not supported")
            return []

        with open(file_path, encoding="utf-8") as f:
            logger.info(f"Reading {str(file_path)}")

            if file_path.suffix == ".json":
                json_obj = json.loads(f.read())
                try:
                    document = Document.from_json(json_obj)
                except Exception as e:
                    raise ValueError(f"Loading JSON failed {e}") from e

            else:
                document = Document(
                    text=f.read(),
                    doc_type=document_type,
                    name=str(file_path),
                    link=str(file_path),
                    timestamp=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    reader=self.name,
                )
            documents.append(document)
        logger.info(f"Loaded {str(file_path)}")
        return documents

    def load_directory(self, dir_path: Path, document_type: str) -> List[Document]:
        """
        Loads documents from a directory and its subdirectories.

        Args:
            dir_path (Path): The path to the directory.
            document_type (str): The type of the documents.

        Returns:
            List[Document]: A list of Document objects representing the loaded documents.

        This function initializes an empty list to store the documents. It then converts the `dir_path` to a string if it is a Path object.

        Next, it loops through each file type specified in `self.file_types`. For each file type, it uses the `glob` module to find all the files in `dir_path` and its subdirectories that match the current file type.

        For each file found, it logs a message indicating that it is reading the file. It then opens the file in read mode with UTF-8 encoding. It creates a `Document` object with the file's contents, the specified `document_type`, the file's name and link, the current timestamp, and the name of the reader. The `Document` object is added to the `documents` list.

        Finally, it logs a success message indicating the number of documents loaded and returns the list of documents.
        """
        # Initialize an empty Dictionary to store the file contents
        documents = []

        # Convert dir_path to string, in case it's a Path object
        dir_path_str = str(dir_path)

        # Loop through each file type
        for file_type in self.file_types:
            # Use glob to find all the files in dir_path and its subdirectories matching the current file_type
            files = glob.glob(f"{dir_path_str}/**/*{file_type}", recursive=True)

            # Loop through each file
            for file in files:
                logger.info(f"Reading {str(file)}")
                with open(file, encoding="utf-8") as f:
                    document = Document(
                        text=f.read(),
                        doc_type=document_type,
                        name=str(file),
                        link=str(file),
                        timestamp=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                        reader=self.name,
                    )

                    documents.append(document)

        logger.info(f"Loaded {len(documents)} documents")
        return documents

        ```

    - 📁 **retrievers/**
      - 📄 **SimpleRetriever.py**

        📄 *File Path*: `./src/vectordb/retrievers/SimpleRetriever.py`

        ```
        """
Simple Retriever. Based on Weaviate's Verba.
https://github.com/weaviate/Verba
"""
from typing_extensions import List, Tuple
from weaviate.client import Client
from weaviate.hybrid import HybridFusion

from src.vectordb.retrievers.interface import Retriever
from src.vectordb.embedders.interface import Embedder
from src.vectordb.chunkers.chunk import Chunk


class SimpleRetriever(Retriever):
    """
    SimpleRetriver that retrieves chunks through hybrid search, no reranking or additional logic.
    """

    def __init__(self):
        super().__init__()
        self.description = "SimpleRetriever uses Hybrid Search to retrieve relevant chunks to the user's query"
        self.name = "SimpleRetriever"

    def retrieve(
        self,
        queries: List[str],
        client: Client,
        embedder: Embedder,
    ) -> Tuple[List[Chunk], str]:
        """Ingest data into Weaviate
        @parameter: queries : List[str] - List of queries
        @parameter: client : Client - Weaviate client
        @parameter: embedder : Embedder - Current selected Embedder
        @returns List[Chunk] - List of retrieved chunks.
        """
        chunk_class = embedder.get_chunk_class()
        needs_vectorization = embedder.get_need_vectorization()
        chunks = []

        for query in queries:
            query_results = (
                client.query.get(
                    class_name=chunk_class,
                    properties=[
                        "text",
                        "doc_name",
                        "chunk_id",
                        "doc_uuid",
                        "doc_type",
                    ],
                )
                .with_additional(properties=["score"])
                .with_autocut(2)
            )

            if needs_vectorization:
                vector = embedder.vectorize_query(query)
                query_results = query_results.with_hybrid(
                    query=query,
                    vector=vector,
                    fusion_type=HybridFusion.RELATIVE_SCORE,
                    properties=[
                        "text",
                    ],
                ).do()

            else:
                query_results = query_results.with_hybrid(
                    query=query,
                    fusion_type=HybridFusion.RELATIVE_SCORE,
                    properties=[
                        "text",
                    ],
                ).do()

            for chunk in query_results["data"]["Get"][chunk_class]:
                chunk_obj = Chunk(
                    chunk["text"],
                    chunk["doc_name"],
                    chunk["doc_type"],
                    chunk["doc_uuid"],
                    chunk["chunk_id"],
                )
                chunk_obj.set_score(chunk["_additional"]["score"])
                chunks.append(chunk_obj)

        sorted_chunks = self.sort_chunks(chunks)

        context = "".join(f"{chunk.text} " for chunk in sorted_chunks)
        return sorted_chunks, context

        ```

      - 📄 **WindowRetriever.py**

        📄 *File Path*: `./src/vectordb/retrievers/WindowRetriever.py`

        ```
        """
Window Retriever. Based one Weaviate's Verba.
https://github.com/weaviate/Verba
"""
from weaviate.client import Client
from weaviate.hybrid import HybridFusion
from typing import List, Tuple

from src.vectordb.embedders.interface import Embedder
from src.vectordb.chunkers.chunk import Chunk
from src.vectordb.retrievers.interface import Retriever


class WindowRetriever(Retriever):
    """
    Window Retriever. uses hybrid search to retrieve relevant chunks and adds their surrounding context
    """

    def __init__(self):
        """
        Initialize a new instance of the WindowRetriever class.

        This method initializes the attributes of the WindowRetriever class, including the description and name.
        It calls the __init__ method of the parent class to ensure proper initialization.

        Parameters:
            None

        Returns:
            None
        """
        super().__init__(
            name="WindowRetriever",
            description="WindowRetriever uses Hybrid Search to retrieve relevant chunks and adds their surrounding context",
            requires_env=[],
            requires_library=[],
        )
        self.description = "WindowRetriever uses Hybrid Search to retrieve relevant chunks and adds their surrounding context"
        self.name = "WindowRetriever"

    def retrieve(
        self,
        queries: List[str],
        client: Client,
        embedder: Embedder,
    ) -> Tuple[List[Chunk], str]:
        """
        Retrieve chunks from Weaviate based on the given queries and return them sorted and with their surrounding context.

        Parameters:
            queries (List[str]): A List of queries to search for chunks.
            client (Client): The Weaviate client used to query the database.
            embedder (Embedder): The embedder used to vectorize the queries.

        Returns:
            Tuple(List[Chunk], str): A Tuple containing a List of sorted chunks and the combined context string.
        """

        chunk_class = embedder.get_chunk_class()
        needs_vectorization = embedder.get_need_vectorization()
        chunks = []

        for query in queries:
            query_results = (
                client.query.get(
                    class_name=chunk_class,
                    properties=[
                        "text",
                        "doc_name",
                        "chunk_id",
                        "doc_uuid",
                        "doc_type",
                    ],
                )
                .with_additional(properties=["score"])
                .with_autocut(2)
            )

            if needs_vectorization:
                vector = embedder.vectorize_query(query)
                query_results = query_results.with_hybrid(
                    query=query,
                    vector=vector,
                    fusion_type=HybridFusion.RELATIVE_SCORE,
                    properties=[
                        "text",
                    ],
                ).do()

            else:
                query_results = query_results.with_hybrid(
                    query=query,
                    fusion_type=HybridFusion.RELATIVE_SCORE,
                    properties=[
                        "text",
                    ],
                ).do()

            for chunk in query_results["data"]["Get"][chunk_class]:
                chunk_obj = Chunk(
                    chunk["text"],
                    chunk["doc_name"],
                    chunk["doc_type"],
                    chunk["doc_uuid"],
                    chunk["chunk_id"],
                )
                chunk_obj.set_score(chunk["_additional"]["score"])
                chunks.append(chunk_obj)

        sorted_chunks = self.sort_chunks(chunks)

        context = self.combine_context(sorted_chunks, client, embedder)

        return sorted_chunks, context

    def combine_context(
        self,
        chunks: List[Chunk],
        client: Client,
        embedder: Embedder,
    ) -> str:
        """
        Combines the context of the given chunks by retrieving and adding surrounding chunks to the map.

        Args:
            chunks (List[Chunk]): A List of chunks to combine context for.
            client (Client): The Weaviate client used to query the database.
            embedder (Embedder): The embedder used to vectorize the queries.

        Returns:
            str: The combined context string of the chunks.

        Description:
            This function takes a List of chunks and combines their context by retrieving and adding surrounding chunks to a map.
            The function iterates over each chunk in the List and checks if its document name is already in the map. If not, it adds an empty Dictionary for that document name.
            Then, for each chunk, it retrieves the chunk ID and creates a range of chunk IDs around it. It iterates over this range and checks if the chunk ID is not already in the map and not in the added chunks Dictionary.
            If the conditions are met, it queries the Weaviate client for the chunk with the given chunk ID and document name. If the query returns a result, it creates a Chunk object from the result and adds it to the added chunks Dictionary.
            Finally, it combines the context of the added chunks and returns it as a string.
        """
        doc_name_map = {}

        context = ""

        for chunk in chunks:
            if chunk.doc_name not in doc_name_map:
                doc_name_map[chunk.doc_name] = {}

            doc_name_map[chunk.doc_name][chunk.chunk_id] = chunk

        window = 2
        for doc, chunk_map in doc_name_map.items():
            added_chunks = {}
            for chunk in chunk_map:
                chunk_id = int(chunk)
                all_chunk_range = list(range(chunk_id - window, chunk_id + window + 1))
                for _range in all_chunk_range:
                    if (
                        _range >= 0
                        and _range not in chunk_map
                        and _range not in added_chunks
                    ):
                        chunk_retrieval_results = (
                            client.query.get(
                                class_name=embedder.get_chunk_class(),
                                properties=[
                                    "text",
                                    "doc_name",
                                    "chunk_id",
                                    "doc_uuid",
                                    "doc_type",
                                ],
                            )
                            .with_where(
                                {
                                    "operator": "And",
                                    "operands": [
                                        {
                                            "path": ["chunk_id"],
                                            "operator": "Equal",
                                            "valueNumber": _range,
                                        },
                                        {
                                            "path": ["doc_name"],
                                            "operator": "Equal",
                                            "valueText": chunk_map[chunk].doc_name,
                                        },
                                    ],
                                }
                            )
                            .with_limit(1)
                            .do()
                        )

                        if (
                            "data" in chunk_retrieval_results
                            and chunk_retrieval_results["data"]["Get"][
                                embedder.get_chunk_class()
                            ]
                        ):
                            chunk_obj = Chunk(
                                chunk_retrieval_results["data"]["Get"][
                                    embedder.get_chunk_class()
                                ][0]["text"],
                                chunk_retrieval_results["data"]["Get"][
                                    embedder.get_chunk_class()
                                ][0]["doc_name"],
                                chunk_retrieval_results["data"]["Get"][
                                    embedder.get_chunk_class()
                                ][0]["doc_type"],
                                chunk_retrieval_results["data"]["Get"][
                                    embedder.get_chunk_class()
                                ][0]["doc_uuid"],
                                chunk_retrieval_results["data"]["Get"][
                                    embedder.get_chunk_class()
                                ][0]["chunk_id"],
                            )
                            added_chunks[str(_range)] = chunk_obj

            for chunk in added_chunks:
                if chunk not in doc_name_map[doc]:
                    doc_name_map[doc][chunk] = added_chunks[chunk]

        for doc in doc_name_map:
            sorted_Dict = {
                k: doc_name_map[doc][k]
                for k in sorted(doc_name_map[doc], key=lambda x: int(x))
            }

            for value in sorted_Dict.values():
                context += value.text

        return context

        ```

      - 📄 **__init__.py**

        📄 *File Path*: `./src/vectordb/retrievers/__init__.py`

        ```
        
        ```

      - 📄 **interface.py**

        📄 *File Path*: `./src/vectordb/retrievers/interface.py`

        ```
        import tiktoken
from loguru import logger
from weaviate.client import Client
from typing_extensions import List, Tuple

from src.vectordb.component import Component
from src.vectordb.chunkers.chunk import Chunk
from src.vectordb.embedders.interface import Embedder


class Retriever(Component):
    """
    Retriever interface for retrieving data from Weaviate.
    """

    def __init__(
        self,
        name: str,
        requires_env: List[str],
        requires_library: List[str],
        description: str,
    ) -> None:
        """
        Initializes a new instance of the class.
        """
        super().__init__(
            description=description,
            name=name,
            requires_env=requires_env,
            requires_library=requires_library,
        )

    def retrieve(
        self,
        queries: List[str],
        client: Client,
        embedder: Embedder,
    ) -> Tuple[List[Chunk], str]:
        """
        Retrieve data from Weaviate using the given queries, client, and embedder.

        Args:
            queries (List[str]): A List of queries to retrieve data from Weaviate.
            client (Client): The Weaviate client used to interact with the Weaviate server.
            embedder (Embedder): The embedder used to embed the retrieved data.

        Returns:
            Tuple[List[Chunk], str]: A Tuple containing a List of retrieved chunks and the context string.

        Raises:
            NotImplementedError: If the load method is not implemented by a subclass.
        """

        raise NotImplementedError("load method must be implemented by a subclass.")

    def sort_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Sorts a List of chunks based on the doc_uuid and chunk_id.

        Args:
            chunks (List[Chunk]): A List of Chunk objects to be sorted.

        Returns:
            List[Chunk]: A sorted List of Chunk objects.
        """
        return sorted(chunks, key=lambda chunk: (chunk.doc_uuid, int(chunk.chunk_id)))

    def cutoff_text(self, text: str, content_length: int) -> str:
        """
        Cuts off the input text to a specified content length in tokens.

        Args:
            text (str): The input text to be cut off.
            content_length (int): The maximum number of tokens in the output text.

        Returns:
            str: The cut off text if the input text exceeds the content length, otherwise the input text.

        Raises:
            None

        Example:
            cutoff_text("This is a long text that needs to be cut off.", 10)
            # Output: "This is a long t..."
        """
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

        # Tokenize the input text
        encoded_tokens = encoding.encode(text, disallowed_special=())

        # Check if we need to truncate
        if len(encoded_tokens) > content_length:
            encoded_tokens = encoded_tokens[:content_length]
            truncated_text = encoding.decode(encoded_tokens)
            logger.info(f"Truncated Context to {content_length} tokens")
            return truncated_text
        else:
            logger.info(f"Retrieved Context of {len(encoded_tokens)} tokens")
            return text

        ```

      - 📄 **manager.py**

        📄 *File Path*: `./src/vectordb/retrievers/manager.py`

        ```
        """
Retriever Manager. Based on Weaviate's Verba.
https://github.com/weaviate/Verba
"""

from typing import List, Tuple, Dict
from weaviate.client import Client
from loguru import logger
from src.vectordb.retrievers.interface import Retriever
from src.vectordb.embedders.interface import Embedder
from src.vectordb.chunkers.chunk import Chunk
from src.text_generators.interface import Generator
from src.vectordb.retrievers.SimpleRetriever import SimpleRetriever
from src.vectordb.retrievers.WindowRetriever import WindowRetriever


class RetrieverManager:
    """
    RetrieverManager class for managing retrievers.

    Attributes:
        retrievers (Dict[str, Retriever]): Dictionary of retrievers.
        selected_retriever (Retriever): Selected retriever for retrieval operations.
    """

    def __init__(self):
        """
        Initializes the RetrieverManager with two retrievers, WindowRetriever and SimpleRetriever, and sets the selected retriever to WindowRetriever.
        """
        self.retrievers: Dict[str, Retriever] = {
            "WindowRetriever": WindowRetriever(),
            "SimpleRetriever": SimpleRetriever(),
        }
        self.selected_retriever: Retriever = self.retrievers["WindowRetriever"]

    def retrieve(
        self,
        queries: List[str],
        client: Client,
        embedder: Embedder,
        generator: Generator,
    ) -> Tuple[List[Chunk], str]:
        """
        Retrieves chunks and managed context using the selected retriever.

        Args:
            queries (List[str]): List of queries to retrieve chunks for.
            client (Client): Client object for making API requests.
            embedder (Embedder): Embedder object for generating embeddings.
            generator (Generator): Generator object for managing context window.

        Returns:
            Tuple[List[Chunk], str]: Tuple containing List of chunks and managed context.
        """
        chunks, context = self.selected_retriever.retrieve(queries, client, embedder)
        managed_context = self.selected_retriever.cutoff_text(
            context, generator.context_window
        )
        return chunks, managed_context

    def set_retriever(self, retriever: str) -> bool:
        """
        Set the selected retriever for retrieval operations.

        Args:
            retriever (str): The name of the retriever to set.

        Returns:
            bool: True if the retriever is found and set successfully, False otherwise.
        """
        if retriever in self.retrievers:
            self.selected_retriever = self.retrievers[retriever]
            return True
        else:
            logger.warning(f"Retriever {retriever} not found")
            return False

    def get_retrievers(self) -> Dict[str, Retriever]:
        """
        Returns the Dictionary of retrievers.
        """
        return self.retrievers

        ```

      - 📄 **simple_retriver.py**

        📄 *File Path*: `./src/vectordb/retrievers/simple_retriver.py`

        ```
        """
Simple Retriever. Based on Weaviate's Verba.
https://github.com/weaviate/Verba
"""
from typing_extensions import List, Tuple
from weaviate.client import Client
from weaviate.hybrid import HybridFusion

from src.vectordb.retrievers.interface import Retriever
from src.vectordb.embedders.interface import Embedder
from src.vectordb.chunkers.chunk import Chunk


class SimpleRetriever(Retriever):
    """
    SimpleRetriver that retrieves chunks through hybrid search, no reranking or additional logic.
    """

    def __init__(self):
        super().__init__()
        self.description = "SimpleRetriever uses Hybrid Search to retrieve relevant chunks to the user's query"
        self.name = "SimpleRetriever"

    def retrieve(
        self,
        queries: List[str],
        client: Client,
        embedder: Embedder,
    ) -> Tuple[List[Chunk], str]:
        """Ingest data into Weaviate
        @parameter: queries : List[str] - List of queries
        @parameter: client : Client - Weaviate client
        @parameter: embedder : Embedder - Current selected Embedder
        @returns List[Chunk] - List of retrieved chunks.
        """
        chunk_class = embedder.get_chunk_class()
        needs_vectorization = embedder.get_need_vectorization()
        chunks = []

        for query in queries:
            query_results = (
                client.query.get(
                    class_name=chunk_class,
                    properties=[
                        "text",
                        "doc_name",
                        "chunk_id",
                        "doc_uuid",
                        "doc_type",
                    ],
                )
                .with_additional(properties=["score"])
                .with_autocut(2)
            )

            if needs_vectorization:
                vector = embedder.vectorize_query(query)
                query_results = query_results.with_hybrid(
                    query=query,
                    vector=vector,
                    fusion_type=HybridFusion.RELATIVE_SCORE,
                    properties=[
                        "text",
                    ],
                ).do()

            else:
                query_results = query_results.with_hybrid(
                    query=query,
                    fusion_type=HybridFusion.RELATIVE_SCORE,
                    properties=[
                        "text",
                    ],
                ).do()

            for chunk in query_results["data"]["Get"][chunk_class]:
                chunk_obj = Chunk(
                    chunk["text"],
                    chunk["doc_name"],
                    chunk["doc_type"],
                    chunk["doc_uuid"],
                    chunk["chunk_id"],
                )
                chunk_obj.set_score(chunk["_additional"]["score"])
                chunks.append(chunk_obj)

        sorted_chunks = self.sort_chunks(chunks)

        context = "".join(f"{chunk.text} " for chunk in sorted_chunks)
        return sorted_chunks, context

        ```

      - 📁 **todo/**
        - 📄 **window_retriver.py**

          📄 *File Path*: `./src/vectordb/retrievers/todo/window_retriver.py`

          ```
          """
Window Retriever. Based one Weaviate's Verba.
https://github.com/weaviate/Verba
"""
from weaviate.client import Client
from weaviate.hybrid import HybridFusion
from typing import List, Tuple

from src.vectordb.embedders.interface import Embedder
from src.vectordb.chunkers.chunk import Chunk
from src.vectordb.retrievers.interface import Retriever


class WindowRetriever(Retriever):
    """
    Window Retriever. uses hybrid search to retrieve relevant chunks and adds their surrounding context
    """

    def __init__(self):
        """
        Initialize a new instance of the WindowRetriever class.

        This method initializes the attributes of the WindowRetriever class, including the description and name.
        It calls the __init__ method of the parent class to ensure proper initialization.

        Parameters:
            None

        Returns:
            None
        """
        super().__init__(
            name="WindowRetriever",
            description="WindowRetriever uses Hybrid Search to retrieve relevant chunks and adds their surrounding context",
            requires_env=[],
            requires_library=[],
        )
        self.description = "WindowRetriever uses Hybrid Search to retrieve relevant chunks and adds their surrounding context"
        self.name = "WindowRetriever"

    def retrieve(
        self,
        queries: List[str],
        client: Client,
        embedder: Embedder,
    ) -> Tuple[List[Chunk], str]:
        """
        Retrieve chunks from Weaviate based on the given queries and return them sorted and with their surrounding context.

        Parameters:
            queries (List[str]): A List of queries to search for chunks.
            client (Client): The Weaviate client used to query the database.
            embedder (Embedder): The embedder used to vectorize the queries.

        Returns:
            Tuple(List[Chunk], str): A Tuple containing a List of sorted chunks and the combined context string.
        """

        chunk_class = embedder.get_chunk_class()
        needs_vectorization = embedder.get_need_vectorization()
        chunks = []

        for query in queries:
            query_results = (
                client.query.get(
                    class_name=chunk_class,
                    properties=[
                        "text",
                        "doc_name",
                        "chunk_id",
                        "doc_uuid",
                        "doc_type",
                    ],
                )
                .with_additional(properties=["score"])
                .with_autocut(2)
            )

            if needs_vectorization:
                vector = embedder.vectorize_query(query)
                query_results = query_results.with_hybrid(
                    query=query,
                    vector=vector,
                    fusion_type=HybridFusion.RELATIVE_SCORE,
                    properties=[
                        "text",
                    ],
                ).do()

            else:
                query_results = query_results.with_hybrid(
                    query=query,
                    fusion_type=HybridFusion.RELATIVE_SCORE,
                    properties=[
                        "text",
                    ],
                ).do()

            for chunk in query_results["data"]["Get"][chunk_class]:
                chunk_obj = Chunk(
                    chunk["text"],
                    chunk["doc_name"],
                    chunk["doc_type"],
                    chunk["doc_uuid"],
                    chunk["chunk_id"],
                )
                chunk_obj.set_score(chunk["_additional"]["score"])
                chunks.append(chunk_obj)

        sorted_chunks = self.sort_chunks(chunks)

        context = self.combine_context(sorted_chunks, client, embedder)

        return sorted_chunks, context

    def combine_context(
        self,
        chunks: List[Chunk],
        client: Client,
        embedder: Embedder,
    ) -> str:
        """
        Combines the context of the given chunks by retrieving and adding surrounding chunks to the map.

        Args:
            chunks (List[Chunk]): A List of chunks to combine context for.
            client (Client): The Weaviate client used to query the database.
            embedder (Embedder): The embedder used to vectorize the queries.

        Returns:
            str: The combined context string of the chunks.

        Description:
            This function takes a List of chunks and combines their context by retrieving and adding surrounding chunks to a map.
            The function iterates over each chunk in the List and checks if its document name is already in the map. If not, it adds an empty Dictionary for that document name.
            Then, for each chunk, it retrieves the chunk ID and creates a range of chunk IDs around it. It iterates over this range and checks if the chunk ID is not already in the map and not in the added chunks Dictionary.
            If the conditions are met, it queries the Weaviate client for the chunk with the given chunk ID and document name. If the query returns a result, it creates a Chunk object from the result and adds it to the added chunks Dictionary.
            Finally, it combines the context of the added chunks and returns it as a string.
        """
        doc_name_map = {}

        context = ""

        for chunk in chunks:
            if chunk.doc_name not in doc_name_map:
                doc_name_map[chunk.doc_name] = {}

            doc_name_map[chunk.doc_name][chunk.chunk_id] = chunk

        window = 2
        for doc, chunk_map in doc_name_map.items():
            added_chunks = {}
            for chunk in chunk_map:
                chunk_id = int(chunk)
                all_chunk_range = list(range(chunk_id - window, chunk_id + window + 1))
                for _range in all_chunk_range:
                    if (
                        _range >= 0
                        and _range not in chunk_map
                        and _range not in added_chunks
                    ):
                        chunk_retrieval_results = (
                            client.query.get(
                                class_name=embedder.get_chunk_class(),
                                properties=[
                                    "text",
                                    "doc_name",
                                    "chunk_id",
                                    "doc_uuid",
                                    "doc_type",
                                ],
                            )
                            .with_where(
                                {
                                    "operator": "And",
                                    "operands": [
                                        {
                                            "path": ["chunk_id"],
                                            "operator": "Equal",
                                            "valueNumber": _range,
                                        },
                                        {
                                            "path": ["doc_name"],
                                            "operator": "Equal",
                                            "valueText": chunk_map[chunk].doc_name,
                                        },
                                    ],
                                }
                            )
                            .with_limit(1)
                            .do()
                        )

                        if (
                            "data" in chunk_retrieval_results
                            and chunk_retrieval_results["data"]["Get"][
                                embedder.get_chunk_class()
                            ]
                        ):
                            chunk_obj = Chunk(
                                chunk_retrieval_results["data"]["Get"][
                                    embedder.get_chunk_class()
                                ][0]["text"],
                                chunk_retrieval_results["data"]["Get"][
                                    embedder.get_chunk_class()
                                ][0]["doc_name"],
                                chunk_retrieval_results["data"]["Get"][
                                    embedder.get_chunk_class()
                                ][0]["doc_type"],
                                chunk_retrieval_results["data"]["Get"][
                                    embedder.get_chunk_class()
                                ][0]["doc_uuid"],
                                chunk_retrieval_results["data"]["Get"][
                                    embedder.get_chunk_class()
                                ][0]["chunk_id"],
                            )
                            added_chunks[str(_range)] = chunk_obj

            for chunk in added_chunks:
                if chunk not in doc_name_map[doc]:
                    doc_name_map[doc][chunk] = added_chunks[chunk]

        for doc in doc_name_map:
            sorted_Dict = {
                k: doc_name_map[doc][k]
                for k in sorted(doc_name_map[doc], key=lambda x: int(x))
            }

            for value in sorted_Dict.values():
                context += value.text

        return context

          ```

    - 📁 **schema/**
      - 📄 **__init__.py**

        📄 *File Path*: `./src/vectordb/schema/__init__.py`

        ```
        
        ```

      - 📄 **schema_generator.py**

        📄 *File Path*: `./src/vectordb/schema/schema_generator.py`

        ```
        """
Schema Generator. Based on Weaviate's Verba.
https://github.com/weaviate/Verba
"""
import re
import os
from dotenv import load_dotenv
from weaviate import Client
from loguru import logger
from typing_extensions import List, Dict, Optional, Tuple

load_dotenv()

VECTORIZERS = {
    "text2vec-openai",
    "text2vec-cohere",
}  # Needs to match with Weaviate modules
EMBEDDINGS = {"MiniLM"}  # Custom Vectors


def strip_non_letters(s: str):
    """
    Replaces all non-alphanumeric characters in a string with underscores.

    Parameters:
        s (str): The input string.

    Returns:
        str: The modified string with non-alphanumeric characters replaced by underscores.
    """
    return re.sub(r"[^a-zA-Z0-9]", "_", s)


def verify_vectorizer(
    schema: Dict, vectorizer: str, skip_properties: Optional[List[str]] = None
) -> Dict:
    """
    Verify the vectorizer and update the schema accordingly.

    Args:
        schema (Dict): The schema to be modified.
        vectorizer (str): The name of the vectorizer.
        skip_properties (Optional[List[str]], optional): The list of properties to skip. Defaults to None.

    Returns:
        Dict: The modified schema.

    Raises:
        ValueError: If the `AZURE_OPENAI_RESOURCE_NAME` and `AZURE_OPENAI_EMBEDDING_MODEL` environment variables are not set when using the Azure OpenAI vectorizer.

    Description:
        This function verifies the vectorizer and updates the schema accordingly. It checks if the vectorizer is in the list of supported vectorizers (`VECTORIZERS`) and updates the schema accordingly. If the vectorizer is not in the list, it checks if it is in the list of supported embeddings (`EMBEDDINGS`). If the vectorizer is not None, it logs a warning message.

        If the vectorizer is the Azure OpenAI vectorizer and the `OPENAI_API_TYPE` environment variable is set to "azure", it checks if the `AZURE_OPENAI_RESOURCE_NAME` and `AZURE_OPENAI_EMBEDDING_MODEL` environment variables are set. If they are not set, it raises a `ValueError` with a specific error message.

        If the vectorizer is the Azure OpenAI vectorizer and the environment variables are set, it creates a `vectorizer_config` dictionary with the deployment ID and resource name. It then updates the schema by setting the vectorizer and module config.

        If any properties in the schema need to be skipped, it updates the module config for the vectorizer to skip the properties and set `vectorizePropertyName` to False.
    """
    if skip_properties is None:
        skip_properties = []
    modified_schema = schema.copy()

    # adding specific config for Azure OpenAI
    vectorizer_config = None
    if os.getenv("OPENAI_API_TYPE") == "azure" and vectorizer == "text2vec-openai":
        resource_name = os.getenv("AZURE_OPENAI_RESOURCE_NAME")
        model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
        if resource_name is None or model is None:
            raise ValueError(
                "AZURE_OPENAI_RESOURCE_NAME and AZURE_OPENAI_EMBEDDING_MODEL should be set when OPENAI_API_TYPE is azure. Resource name is XXX in http://XXX.openai.azure.com"
            )
        vectorizer_config = {
            "text2vec-openai": {"deploymentId": model, "resource_name": resource_name}
        }

    # Verify Vectorizer
    if vectorizer in VECTORIZERS:
        modified_schema["classes"][0]["vectorizer"] = vectorizer
        if vectorizer_config is not None:
            modified_schema["classes"][0]["module_config"] = vectorizer_config
        for prop in modified_schema["classes"][0]["properties"]:
            if prop["name"] in skip_properties:
                module_config = {
                    vectorizer: {
                        "skip": True,
                        "vectorizePropertyName": False,
                    }
                }
                prop["module_config"] = module_config
    elif vectorizer in EMBEDDINGS:
        pass
    elif vectorizer is not None:
        logger.warning(f"Could not find matching vectorizer: {vectorizer}")

    return modified_schema


def add_suffix(schema: Dict, vectorizer: str) -> Tuple[Dict, str]:
    """
    A function that adds a suffix to the class property of the schema based on the vectorizer provided.

    Args:
        schema (Dict): The schema to be modified.
        vectorizer (str): The name of the vectorizer.

    Returns:
        Tuple[Dict, str]: A tuple containing the modified schema and the updated class property.
    """
    modified_schema = schema.copy()
    # Verify Vectorizer and add suffix
    modified_schema["classes"][0]["class"] = (
        modified_schema["classes"][0]["class"] + "_" + strip_non_letters(vectorizer)
    )
    return modified_schema, modified_schema["classes"][0]["class"]


def reset_schemas(
    client: Optional[Client] = None,
    vectorizer: Optional[str] = None,
):
    """
    Reset the schemas for a given client and vectorizer.

    Args:
        client (Optional[Client]): The client object used to interact with the schemas. Defaults to None.
        vectorizer (Optional[str]): The name of the vectorizer. Defaults to None.

    Returns:
        None
    """
    if not client or not vectorizer:
        return
    doc_name = f"Document_{strip_non_letters(vectorizer)}"
    chunk_name = f"Chunk_{strip_non_letters(vectorizer)}"
    cache_name = f"Cache_{strip_non_letters(vectorizer)}"

    client.schema.delete_class(doc_name)
    client.schema.delete_class(chunk_name)
    client.schema.delete_class(cache_name)


def init_schemas(
    client: Optional[Client] = None,
    vectorizer: Optional[str] = None,
    force: bool = False,
    check: bool = False,
) -> bool:
    """
    Initializes the schemas for a given client and vectorizer.

    Args:
        client (Optional[Client]): The client object used to interact with the schemas. Defaults to None.
        vectorizer (Optional[str]): The name of the vectorizer. Defaults to None.
        force (bool, optional): Whether to force the initialization even if the schemas already exist. Defaults to False.
        check (bool, optional): Whether to check if the schemas already exist before initializing. Defaults to False.

    Returns:
        bool: True if the schemas are successfully initialized, False otherwise.

    Raises:
        ValueError: If the schema initialization fails.

    """
    if not client or vectorizer:
        return False
    try:
        init_documents(client, vectorizer, force, check)
        init_cache(client, vectorizer, force, check)
        init_suggestion(client, force, check)
        return True
    except ValueError as e:
        logger.warning(f"Schema initialization failed {str(e)}")
        return False


def init_documents(
    client: Client,
    vectorizer: Optional[str] = None,
    force: bool = False,
    check: bool = False,
) -> Tuple[Dict, Dict]:
    """
    Initializes the schemas for a given client and vectorizer.

    Args:
        client (Client): The client object used to interact with the schemas.
        vectorizer (Optional[str], optional): The name of the vectorizer. Defaults to None.
        force (bool, optional): Whether to force the initialization even if the schemas already exist. Defaults to False.
        check (bool, optional): Whether to check if the schemas already exist before initializing. Defaults to False.

    Returns:
        Tuple[Dict, Dict]: A tuple containing the document schema and the chunk schema.

    Raises:
        ValueError: If the schema initialization fails.

    This function initializes the schemas for a given client and vectorizer. It creates two schema classes: "Document" and "Chunk". The "Document" class has properties such as "text", "doc_name", "doc_type", "doc_link", "timestamp", and "chunk_count". The "Chunk" class has properties such as "text", "doc_name", and "chunk_id". The function verifies the vectorizer and adds a suffix to the schema names. If the schema classes already exist, the function prompts the user to delete them. If the user agrees, the function deletes the schemas and creates new ones. Finally, the function returns the document schema and chunk schema.
    """
    if not vectorizer:
        return {}, {}
    SCHEMA_CHUNK = {
        "classes": [
            {
                "class": "Chunk",
                "description": "Chunks of Documentations",
                "properties": [
                    {
                        "name": "text",
                        "dataType": ["text"],
                        "description": "Content of the document",
                    },
                    {
                        "name": "doc_name",
                        "dataType": ["text"],
                        "description": "Document name",
                    },
                    {
                        # Skip
                        "name": "doc_type",
                        "dataType": ["text"],
                        "description": "Document type",
                    },
                    {
                        # Skip
                        "name": "doc_uuid",
                        "dataType": ["text"],
                        "description": "Document UUID",
                    },
                    {
                        # Skip
                        "name": "chunk_id",
                        "dataType": ["number"],
                        "description": "Document chunk from the whole document",
                    },
                ],
            }
        ]
    }
    SCHEMA_DOCUMENT = {
        "classes": [
            {
                "class": "Document",
                "description": "Documentation",
                "properties": [
                    {
                        "name": "text",
                        "dataType": ["text"],
                        "description": "Content of the document",
                    },
                    {
                        "name": "doc_name",
                        "dataType": ["text"],
                        "description": "Document name",
                    },
                    {
                        "name": "doc_type",
                        "dataType": ["text"],
                        "description": "Document type",
                    },
                    {
                        "name": "doc_link",
                        "dataType": ["text"],
                        "description": "Link to document",
                    },
                    {
                        "name": "timestamp",
                        "dataType": ["text"],
                        "description": "Timestamp of document",
                    },
                    {
                        "name": "chunk_count",
                        "dataType": ["number"],
                        "description": "Number of chunks",
                    },
                ],
            }
        ]
    }
    # Verify Vectorizer
    chunk_schema = verify_vectorizer(
        SCHEMA_CHUNK,
        vectorizer,
        ["doc_type", "doc_uuid", "chunk_id"],
    )

    # Add Suffix
    document_schema, document_name = add_suffix(SCHEMA_DOCUMENT, vectorizer)
    chunk_schema, chunk_name = add_suffix(chunk_schema, vectorizer)

    if client.schema.exists(document_name):
        if check:
            return document_schema, chunk_schema
        if not force:
            user_input = input(
                f"{document_name} class already exists, do you want to delete it? (y/n): "
            )
        else:
            user_input = "y"
        if user_input.strip().lower() == "y":
            client.schema.delete_class(document_name)
            client.schema.delete_class(chunk_name)
            client.schema.create(document_schema)
            client.schema.create(chunk_schema)
            logger.info(f"{document_name} and {chunk_name} schemas created")
        else:
            logger.warning(
                f"Skipped deleting {document_name} and {chunk_name} schema, nothing changed"
            )
    else:
        client.schema.create(document_schema)
        client.schema.create(chunk_schema)
        logger.info(f"{document_name} and {chunk_name} schemas created")

    return document_schema, chunk_schema


def init_cache(
    client: Client,
    vectorizer: Optional[str] = None,
    force: bool = False,
    check: bool = False,
) -> Dict:
    """
    Initializes the cache schema for a given client and vectorizer.

    Args:
        client (Client): The client object used to interact with the schemas.
        vectorizer (Optional[str]): The name of the vectorizer. Defaults to None.
        force (bool, optional): Whether to force the initialization even if the schemas already exist. Defaults to False.
        check (bool, optional): Whether to check if the schemas already exist before initializing. Defaults to False.

    Returns:
        Dict: The cache schema.
    """
    if not vectorizer:
        return {}
    SCHEMA_CACHE = {
        "classes": [
            {
                "class": "Cache",
                "description": "Cache of Documentations and their queries",
                "properties": [
                    {
                        "name": "query",
                        "dataType": ["text"],
                        "description": "Query",
                    },
                    {
                        # Skip
                        "name": "system",
                        "dataType": ["text"],
                        "description": "System message",
                    },
                ],
            }
        ]
    }

    # Verify Vectorizer
    cache_schema = verify_vectorizer(
        SCHEMA_CACHE,
        vectorizer,
        ["system", "results"],
    )

    # Add Suffix
    cache_schema, cache_name = add_suffix(cache_schema, vectorizer)

    if client.schema.exists(cache_name):
        if check:
            return cache_schema
        if not force:
            user_input = input(
                f"{cache_name} class already exists, do you want to delete it? (y/n): "
            )
        else:
            user_input = "y"
        if user_input.strip().lower() == "y":
            client.schema.delete_class(cache_name)
            client.schema.create(cache_schema)
            logger.info(f"{cache_name} schema created")
        else:
            logger.warning(f"Skipped deleting {cache_name} schema, nothing changed")
    else:
        client.schema.create(cache_schema)
        logger.info(f"{cache_name} schema created")

    return cache_schema


def init_suggestion(client: Client, force: bool = False, check: bool = False) -> Dict:
    """
    Initializes the schema for suggestions based on the client, force flag, and check flag.

    Args:
        client (Client): The client object used to interact with the schemas.
        force (bool, optional): Whether to force the initialization even if the schema exists. Defaults to False.
        check (bool, optional): Whether to check if the schema already exists before initializing. Defaults to False.

    Returns:
        Dict: The schema for suggestions.

    """
    SCHEMA_SUGGESTION = {
        "classes": [
            {
                "class": "Suggestion",
                "description": "List of possible prompts",
                "properties": [
                    {
                        "name": "suggestion",
                        "dataType": ["text"],
                        "description": "Query",
                    },
                ],
            }
        ]
    }

    suggestion_schema = SCHEMA_SUGGESTION
    suggestion_name = "Suggestion"

    if client.schema.exists(suggestion_name):
        if check:
            return suggestion_schema
        if not force:
            user_input = input(
                f"{suggestion_name} class already exists, do you want to delete it? (y/n): "
            )
        else:
            user_input = "y"
        if user_input.strip().lower() == "y":
            client.schema.delete_class(suggestion_name)
            client.schema.create(suggestion_schema)
            logger.info(f"{suggestion_name} schema created")
        else:
            logger.warning(
                f"Skipped deleting {suggestion_name} schema, nothing changed"
            )
    else:
        client.schema.create(suggestion_schema)
        logger.info(f"{suggestion_name} schema created")

    return suggestion_schema

        ```

    - 📁 **splitter/**
      - 📄 **interface.py**

        📄 *File Path*: `./src/vectordb/splitter/interface.py`

        ```
        class Chunker(Component):
    """
    Interface for Verba Chunking.
    """

    def __init__(self):
        super().__init__()
        self.input_form = InputForm.CHUNKER.value  # Default for all Chunkers
        self.default_units = 100
        self.default_overlap = 50

    def chunk(
        self, documents: list[Document], units: int, overlap: int
    ) -> list[Document]:
        """Chunk verba documents into chunks based on units and overlap.

        @parameter: documents : list[Document] - List of Verba documents
        @parameter: units : int - How many units per chunk (words, sentences, etc.)
        @parameter: overlap : int - How much overlap between the chunks
        @returns list[str] - List of documents that contain the chunks.
        """
        raise NotImplementedError("chunk method must be implemented by a subclass.")

        ```

      - 📄 **manager.py**

        📄 *File Path*: `./src/vectordb/splitter/manager.py`

        ```
        class ChunkerManager:
    def __init__(self):
        self.chunker: dict[str, Chunker] = {
            "TokenChunker": TokenChunker(),
            "WordChunker": WordChunker(),
            "SentenceChunker": SentenceChunker(),
        }
        self.selected_chunker: Chunker = self.chunker["TokenChunker"]

    def chunk(
        self, documents: list[Document], units: int, overlap: int
    ) -> list[Document]:
        """Chunk verba documents into chunks based on n and overlap.

        @parameter: documents : list[Document] - List of Verba documents
        @parameter: units : int - How many units per chunk (words, sentences, etc.)
        @parameter: overlap : int - How much overlap between the chunks
        @returns list[str] - List of documents that contain the chunks.
        """
        chunked_docs = self.selected_chunker.chunk(documents, units, overlap)
        msg.good("Chunking completed")
        if self.check_chunks(chunked_docs):
            return chunked_docs
        return []

    def set_chunker(self, chunker: str) -> bool:
        if chunker in self.chunker:
            self.selected_chunker = self.chunker[chunker]
            return True
        else:
            msg.warn(f"Chunker {chunker} not found")
            return False

    def get_chunkers(self) -> dict[str, Chunker]:
        return self.chunker

    def check_chunks(self, documents: list[Document]) -> bool:
        """Checks token count of chunks which are hardcapped to 1000 tokens per chunk
        @parameter: documents : list[Document] - List of Verba documents
        @returns bool - Whether the chunks are within the token range.
        """
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

        for document in documents:
            chunks = document.chunks
            for chunk in chunks:
                tokens = encoding.encode(chunk.text, disallowed_special=())
                chunk.set_tokens(tokens)
                if len(tokens) > 1000:
                    raise Exception(
                        "Chunk detected with more than 1000 tokens which exceeds the maximum size. Please reduce size of your chunk."
                    )

        return True

        ```

      - 📄 **sentence_splitter.py**

        📄 *File Path*: `./src/vectordb/splitter/sentence_splitter.py`

        ```
        from src.data_models.document import Document
from src.data_models.chunk import Chunk


class SentenceChunker(Chunker):
    """
    SentenceChunker for Verba built with spaCy.
    """

    def __init__(self):
        super().__init__()
        self.name = "WordChunker"
        self.requires_library = ["spacy"]
        self.default_units = 3
        self.default_overlap = 2
        self.description = "Chunk documents by sentences. You can specify how many sentences should overlap between chunks to improve retrieval."
        try:
            self.nlp = spacy.blank("en")
            self.nlp.add_pipe("sentencizer")
        except:
            self.nlp = None

    def chunk(
        self, documents: list[Document], units: int, overlap: int
    ) -> list[Document]:
        """Chunk verba documents into chunks based on units and overlap
        @parameter: documents : list[Document] - List of Verba documents
        @parameter: units : int - How many units per chunk (words, sentences, etc.)
        @parameter: overlap : int - How much overlap between the chunks
        @returns list[str] - List of documents that contain the chunks.
        """
        for document in tqdm(
            documents, total=len(documents), desc="Chunking documents"
        ):
            # Skip if document already contains chunks
            if len(document.chunks) > 0:
                continue

            doc = list(self.nlp(document.text).sents)

            if units > len(doc) or units < 1:
                msg.warn(
                    f"Unit value either exceeds length of actual document or is below 1 ({units}/{len(doc)})"
                )
                continue

            if overlap >= units:
                msg.warn(
                    f"Overlap value is greater than unit (Units {units}/ Overlap {overlap})"
                )
                continue

            i = 0
            split_id_counter = 0
            while i < len(doc):
                # Overlap
                start_i = i
                end_i = i + units
                if end_i > len(doc):
                    end_i = len(doc)  # Adjust for the last chunk

                text = ""
                for sent in doc[start_i:end_i]:
                    text += sent.text

                doc_chunk = Chunk(
                    text=text,
                    doc_name=document.name,
                    doc_type=document.type,
                    chunk_id=split_id_counter,
                )
                document.chunks.append(doc_chunk)
                split_id_counter += 1

                # Exit loop if this was the last possible chunk
                if end_i == len(doc):
                    break

                i += units - overlap  # Step forward, considering overlap

        return documents

        ```

      - 📄 **tiktoken_chunker.py**

        📄 *File Path*: `./src/vectordb/splitter/tiktoken_chunker.py`

        ```
        class TokenChunker(Chunker):
    """
    TokenChunker for Verba built with tiktoken.
    """

    def __init__(self):
        super().__init__()
        self.name = "TokenChunker"
        self.requires_library = ["tiktoken"]
        self.default_units = 250
        self.default_overlap = 50
        self.description = "Chunk documents by tokens powered by tiktoken. You can specify how many tokens should overlap between chunks to improve retrieval."
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def chunk(
        self, documents: list[Document], units: int, overlap: int
    ) -> list[Document]:
        """Chunk verba documents into chunks based on units and overlap
        @parameter: documents : list[Document] - List of Verba documents
        @parameter: units : int - How many units per chunk (words, sentences, etc.)
        @parameter: overlap : int - How much overlap between the chunks
        @returns list[str] - List of documents that contain the chunks.
        """
        for document in tqdm(
            documents, total=len(documents), desc="Chunking documents"
        ):
            # Skip if document already contains chunks
            if len(document.chunks) > 0:
                continue

            encoded_tokens = self.encoding.encode(document.text, disallowed_special=())

            if units > len(encoded_tokens) or units < 1:
                doc_chunk = Chunk(
                    text=document.text,
                    doc_name=document.name,
                    doc_type=document.type,
                    chunk_id=0,
                )

            if overlap >= units:
                msg.warn(
                    f"Overlap value is greater than unit (Units {units}/ Overlap {overlap})"
                )
                continue

            i = 0
            split_id_counter = 0
            while i < len(encoded_tokens):
                # Overlap
                start_i = i
                end_i = min(i + units, len(encoded_tokens))

                chunk_tokens = encoded_tokens[start_i:end_i]
                chunk_text = self.encoding.decode(chunk_tokens)

                doc_chunk = Chunk(
                    text=chunk_text,
                    doc_name=document.name,
                    doc_type=document.type,
                    chunk_id=split_id_counter,
                )
                document.chunks.append(doc_chunk)
                split_id_counter += 1

                # Exit loop if this was the last possible chunk
                if end_i == len(encoded_tokens):
                    break

                i += units - overlap  # Step forward, considering overlap

        return documents

        ```

      - 📄 **work_splitter.py**

        📄 *File Path*: `./src/vectordb/splitter/work_splitter.py`

        ```
        class WordChunker(Chunker):
    """
    WordChunker for Verba built with spaCy.
    """

    def __init__(self):
        super().__init__()
        self.name = "WordChunker"
        self.requires_library = ["spacy"]
        self.default_units = 100
        self.default_overlap = 50
        self.description = "Chunk documents by words. You can specify how many words should overlap between chunks to improve retrieval."
        try:
            self.nlp = spacy.blank("en")
        except:
            self.nlp = None

    def chunk(
        self, documents: list[Document], units: int, overlap: int
    ) -> list[Document]:
        """Chunk verba documents into chunks based on units and overlap
        @parameter: documents : list[Document] - List of Verba documents
        @parameter: units : int - How many units per chunk (words, sentences, etc.)
        @parameter: overlap : int - How much overlap between the chunks
        @returns list[str] - List of documents that contain the chunks.
        """
        for document in tqdm(
            documents, total=len(documents), desc="Chunking documents"
        ):
            # Skip if document already contains chunks
            if len(document.chunks) > 0:
                continue

            doc = self.nlp(document.text)

            if units > len(doc) or units < 1:
                doc_chunk = Chunk(
                    text=doc.text,
                    doc_name=document.name,
                    doc_type=document.type,
                    chunk_id=0,
                )

            if overlap >= units:
                msg.warn(
                    f"Overlap value is greater than unit (Units {units}/ Overlap {overlap})"
                )
                continue

            i = 0
            split_id_counter = 0
            while i < len(doc):
                # Overlap
                start_i = i
                end_i = i + units
                if end_i > len(doc):
                    end_i = len(doc)  # Adjust for the last chunk

                doc_chunk = Chunk(
                    text=doc[start_i:end_i].text,
                    doc_name=document.name,
                    doc_type=document.type,
                    chunk_id=split_id_counter,
                )
                document.chunks.append(doc_chunk)
                split_id_counter += 1

                # Exit loop if this was the last possible chunk
                if end_i == len(doc):
                    break

                i += units - overlap  # Step forward, considering overlap

        return documents

        ```

    - 📄 **weaviate_manager.py**

      📄 *File Path*: `./src/vectordb/weaviate_manager.py`

      ```
      from tqdm import tqdm
from src.vectordb.embedders.interface import (
    strip_non_letters,
    EMBEDDINGS,
    VECTORIZERS,
    InputForm,
    Document,
    Client,
    Component
)
from src.utils.logging import get_logger


logger = get_logger()


class Embedder(Component):
    """
    Interface for Embedding.
    """

    def __init__(self):
        super().__init__()
        self.input_form = InputForm.TEXT.value  # Default for all Embedders
        self.vectorizer = ""

    def embed(self, documents: list[Document], client: Client, batch_size: int = 100) -> bool:
        """Embed verba documents and its chunks to Weaviate
        @parameter: documents : list[Document] - List of Verba documents
        @parameter: client : Client - Weaviate Client
        @parameter: batch_size : int - Batch Size of Input
        @returns bool - Bool whether the embedding what successful.
        """
        raise NotImplementedError("embed method must be implemented by a subclass.")

    def import_data(
        self,
        documents: list[Document],
        client: Client,
    ) -> bool:
        """Import verba documents and its chunks to Weaviate
        @parameter: documents : list[Document] - List of Verba documents
        @parameter: client : Client - Weaviate Client
        @parameter: batch_size : int - Batch Size of Input
        @returns bool - Bool whether the embedding what successful.
        """
        try:
            if self.vectorizer not in VECTORIZERS and self.vectorizer not in EMBEDDINGS:
                logger.error(f"Vectorizer of {self.name} not found")
                return False

            for i, document in enumerate(documents):
                batches = []
                uuid = ""
                temp_batch = []
                token_counter = 0
                for chunk in document.chunks:
                    if token_counter + len(chunk.tokens) <= 5000:
                        token_counter += len(chunk.tokens)
                        temp_batch.append(chunk)
                    else:
                        batches.append(temp_batch.copy())
                        token_counter = len(chunk.tokens)
                        temp_batch = [chunk]
                if len(temp_batch) > 0:
                    batches.append(temp_batch.copy())
                    token_counter = 0
                    temp_batch = []

                logger.info(
                    f"({i+1}/{len(documents)}) Importing document {document.name} with {len(batches)} batches"
                )

                with client.batch as batch:
                    batch.batch_size = 1
                    properties = {
                        "text": str(document.text),
                        "doc_name": str(document.name),
                        "doc_type": str(document.type),
                        "doc_link": str(document.link),
                        "chunk_count": len(document.chunks),
                        "timestamp": str(document.timestamp),
                    }

                    class_name = "Document_" + strip_non_letters(self.vectorizer)
                    uuid = client.batch.add_data_object(properties, class_name)

                    for chunk in document.chunks:
                        chunk.set_uuid(uuid)

                chunk_count = 0
                for _batch_id, chunk_batch in tqdm(
                    enumerate(batches), total=len(batches), desc="Importing batches"
                ):
                    with client.batch as batch:
                        batch.batch_size = len(chunk_batch)
                        for i, chunk in enumerate(chunk_batch):
                            chunk_count += 1

                            properties = {
                                "text": chunk.text,
                                "doc_name": str(document.name),
                                "doc_uuid": chunk.doc_uuid,
                                "doc_type": chunk.doc_type,
                                "chunk_id": chunk.chunk_id,
                            }
                            class_name = "Chunk_" + strip_non_letters(self.vectorizer)

                            # Check if vector already exists
                            if chunk.vector is None:
                                client.batch.add_data_object(properties, class_name)
                            else:
                                client.batch.add_data_object(
                                    properties, class_name, vector=chunk.vector
                                )

                self.check_document_status(
                    client,
                    uuid,
                    document.name,
                    "Document_" + strip_non_letters(self.vectorizer),
                    "Chunk_" + strip_non_letters(self.vectorizer),
                    len(document.chunks),
                )
            return True
        except Exception as e:
            raise Exception(e)

    def check_document_status(
        self,
        client: Client,
        doc_uuid: str,
        doc_name: str,
        doc_class_name: str,
        chunk_class_name: str,
        chunk_count: int,
    ):
        """Verifies that imported documents and its chunks exist in the database, if not, remove everything that was added and rollback
        @parameter: client : Client - Weaviate Client
        @parameter: doc_uuid : str - Document UUID
        @parameter: doc_name : str - Document name
        @parameter: doc_class_name : str - Class name of Document
        @parameter: chunk_class_name : str - Class name of Chunks
        @parameter: chunk_count : int - Number of expected chunks
        @returns Optional[Exception] - Raises Exceptions if imported fail, will be catched by the manager.
        """
        document = client.data_object.get_by_id(
            doc_uuid,
            class_name=doc_class_name,
        )

        if document is not None:
            results = (
                client.query.get(
                    class_name=chunk_class_name,
                    properties=[
                        "doc_name",
                    ],
                )
                .with_where(
                    {
                        "path": ["doc_uuid"],
                        "operator": "Equal",
                        "valueText": doc_uuid,
                    }
                )
                .with_limit(chunk_count + 1)
                .do()
            )

            if len(results["data"]["Get"][chunk_class_name]) != chunk_count:
                # Rollback if fails
                self.remove_document(client, doc_name, doc_class_name, chunk_class_name)
                raise Exception(
                    f"Chunk mismatch for {doc_uuid} {len(results['data']['Get'][chunk_class_name])} != {chunk_count}"
                )
        else:
            raise Exception(f"Document {doc_uuid} not found {document}")

    def remove_document(
        self, client: Client, doc_name: str, doc_class_name: str, chunk_class_name: str
    ) -> None:
        """Deletes documents and its chunks
        @parameter: client : Client - Weaviate Client
        @parameter: doc_name : str - Document name
        @parameter: doc_class_name : str - Class name of Document
        @parameter: chunk_class_name : str - Class name of Chunks.
        """
        client.batch.delete_objects(
            class_name=doc_class_name,
            where={"path": ["doc_name"], "operator": "Equal", "valueText": doc_name},
        )

        client.batch.delete_objects(
            class_name=chunk_class_name,
            where={"path": ["doc_name"], "operator": "Equal", "valueText": doc_name},
        )

        logger.warning(f"Deleted document {doc_name} and its chunks")

    def remove_document_by_id(self, client: Client, doc_id: str):
        doc_class_name = "Document_" + strip_non_letters(self.vectorizer)
        chunk_class_name = "Chunk_" + strip_non_letters(self.vectorizer)

        client.data_object.delete(uuid=doc_id, class_name=doc_class_name)

        client.batch.delete_objects(
            class_name=chunk_class_name,
            where={"path": ["doc_uuid"], "operator": "Equal", "valueText": doc_id},
        )

        logger.warning(f"Deleted document {doc_id} and its chunks")

    def get_document_class(self) -> str:
        return "Document_" + strip_non_letters(self.vectorizer)

    def get_chunk_class(self) -> str:
        return "Chunk_" + strip_non_letters(self.vectorizer)

    def get_cache_class(self) -> str:
        return "Cache_" + strip_non_letters(self.vectorizer)

    def search_documents(self, client: Client, query: str, doc_type: str) -> list:
        """Search for documents from Weaviate
        @parameter query_string : str - Search query
        @returns list - Document list.
        """
        doc_class_name = "Document_" + strip_non_letters(self.vectorizer)

        if doc_type == "" or doc_type is None:
            query_results = (
                client.query.get(
                    class_name=doc_class_name,
                    properties=["doc_name", "doc_type", "doc_link"],
                )
                .with_bm25(query, properties=["doc_name"])
                .with_additional(properties=["id"])
                .with_limit(100)
                .do()
            )
        else:
            query_results = (
                client.query.get(
                    class_name=doc_class_name,
                    properties=["doc_name", "doc_type", "doc_link"],
                )
                .with_bm25(query, properties=["doc_name"])
                .with_where(
                    {
                        "path": ["doc_type"],
                        "operator": "Equal",
                        "valueText": doc_type,
                    }
                )
                .with_additional(properties=["id"])
                .with_limit(100)
                .do()
            )

        results = query_results["data"]["Get"][doc_class_name]
        return results

    def get_need_vectorization(self) -> bool:
        if self.vectorizer in EMBEDDINGS:
            return True
        return False

    def vectorize_query(self, query: str):
        raise NotImplementedError(
            "vectorize_query method must be implemented by a subclass."
        )

    def conversation_to_query(self, queries: list[str], conversation: dict) -> str:
        query = ""

        if len(conversation) > 1:
            if conversation[-1].type == "system":
                query += conversation[-1].content + " "
            elif conversation[-2].type == "system":
                query += conversation[-2].content + " "

        for _query in queries:
            query += _query + " "

        return query.lower()

    def retrieve_semantic_cache(
        self, client: Client, query: str, dist: float = 0.04
    ) -> str:
        """Retrieve results from semantic cache based on query and distance threshold
        @parameter query - str - User query
        @parameter dist - float - Distance threshold
        @returns Optional[dict] - List of results or None.
        """
        needs_vectorization = self.get_need_vectorization()

        match_results = (
            client.query.get(
                class_name=self.get_cache_class(),
                properties=["query", "system"],
            )
            .with_where(
                {
                    "path": ["query"],
                    "operator": "Equal",
                    "valueText": query,
                }
            )
            .with_limit(1)
        ).do()

        if (
            "data" in match_results
            and len(match_results["data"]["Get"][self.get_cache_class()]) > 0
            and (
                query
                == match_results["data"]["Get"][self.get_cache_class()][0]["query"]
            )
        ):
            logger.success("Direct match from cache")
            return (
                match_results["data"]["Get"][self.get_cache_class()][0]["system"],
                0.0,
            )

        query_results = (
            client.query.get(
                class_name=self.get_cache_class(),
                properties=["query", "system"],
            )
            .with_additional(properties=["distance"])
            .with_limit(1)
        )

        if needs_vectorization:
            vector = self.vectorize_query(query)
            query_results = query_results.with_near_vector(
                content={"vector": vector},
            ).do()

        else:
            query_results = query_results.with_near_text(
                content={"concepts": [query]},
            ).do()

        if "data" not in query_results:
            logger.warning(query_results)
            return None, None

        results = query_results["data"]["Get"][self.get_cache_class()]

        if not results:
            return None, None

        result = results[0]

        if float(result["_additional"]["distance"]) <= dist:
            logger.success("Retrieved similar from cache")
            return result["system"], float(result["_additional"]["distance"])

        else:
            return None, None

    def add_to_semantic_cache(self, client: Client, query: str, system: str):
        """Add results to semantic cache
        @parameter query : str - User query
        @parameter results : list[dict] - Results from Weaviate
        @parameter system : str - System message
        @returns None.
        """
        needs_vectorization = self.get_need_vectorization()

        with client.batch as batch:
            batch.batch_size = 1
            properties = {
                "query": str(query),
                "system": system,
            }
            logger.success("Saved to cache")

            if needs_vectorization:
                vector = self.vectorize_query(query)
                client.batch.add_data_object(
                    properties, self.get_cache_class(), vector=vector
                )
            else:
                client.batch.add_data_object(properties, self.get_cache_class())

      ```

    - 📄 **weaviate_utils.py**

      📄 *File Path*: `./src/vectordb/weaviate_utils.py`

      ```
      import weaviate
from typing import List, Dict
import os

client = weaviate.Client(
    url=os.getenv("WEAVIATE_URL"),
    auth_client_secret=weaviate.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY"))
)

def store_document(document: Dict[str, str]) -> str:
    """
    Store a document in Weaviate.
    
    Args:
    document (Dict[str, str]): A dictionary containing the document data.
    
    Returns:
    str: The UUID of the stored document.
    """
    return client.data_object.create(
        data_object=document,
        class_name="Document"
    )

def retrieve_documents(query: str, limit: int = 5) -> List[Dict[str, str]]:
    """
    Retrieve documents from Weaviate based on a semantic search query.
    
    Args:
    query (str): The search query.
    limit (int): The maximum number of documents to retrieve.
    
    Returns:
    List[Dict[str, str]]: A list of retrieved documents.
    """
    result = (
        client.query
        .get("Document", ["content", "metadata"])
        .with_near_text({"concepts": [query]})
        .with_limit(limit)
        .do()
    )
    return result["data"]["Get"]["Document"]

def retrieve_file_chunks(file_ids: List[str], query: str, limit: int = 5) -> List[Dict[str, str]]:
    """
    Retrieve file chunks from Weaviate based on file IDs and a semantic search query.
    
    Args:
    file_ids (List[str]): List of file IDs to search within.
    query (str): The search query.
    limit (int): The maximum number of chunks to retrieve.
    
    Returns:
    List[Dict[str, str]]: A list of retrieved file chunks.
    """
    result = (
        client.query
        .get("FileChunk", ["content", "metadata"])
        .with_near_text({"concepts": [query]})
        .with_where({
            "path": ["metadata", "file_id"],
            "operator": "In",
            "valueString": file_ids
        })
        .with_limit(limit)
        .do()
    )
    return result["data"]["Get"]["FileChunk"]
      ```

