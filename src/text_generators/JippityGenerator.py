from src.text_generators.interface import Generator, available_generators

class GPT4Generator(Generator):
    """
    GPT4 Generator.
    """

    def __init__(self):
        super().__init__()
        self.name = "GPT4Generator"
        self.description = "Generator using OpenAI's GPT-4-1106-preview model"
        self.requires_library = ["openai"]
        self.requires_env = ["HF_TOKEN"]
        self.streamable = True
        self.model_name = "codellama"
        self.context_window = 10000

    async def generate(
        self,
        queries: List[str],
        context: List[str],
        conversation: Dict = None,
    ) -> str:
        """Generate an answer based on a List of queries and List of contexts, and includes conversational context
        @parameter: queries : List[str] - List of queries
        @parameter: context : List[str] - List of contexts
        @parameter: conversation : Dict - Conversational context
        @returns str - Answer generated by the Generator.
        """
        if conversation is None:
            conversation = {}
        messages = self.prepare_messages(queries, context, conversation)

        try:
            import openai

            openai.api_key = "sk-1234"
            openai.api_base = "https://the-roost-agentartificial.ngrok.dev/code"

            completion = agentartificial_generator(messages)
            system_msg = str(completion["choices"][0]["message"]["content"])

        except Exception:
            raise

        return system_msg

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
            import openai

            openai.api_key = os.getenv("sk-1234")
            openai.base_url = "https://the-roost-agentartificial.ngrok.dev/code"

            completion = agentartificial_generator(messages, streaming=True)

            try:
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
            except StopAsyncIteration:
                pass

        except Exception:
            raise

    def prepare_messages(self, queries: List[str], context: List[str], conversation: Dict[str, str]) -> Dict[str, str]:
        """
        Prepares a List of messages formatted for a Retrieval Augmented Generation chatbot system, including system instructions, previous conversation, and a new user query with context.

        @parameter queries: A List of strings representing the user queries to be answered.
        @parameter context: A List of strings representing the context information provided for the queries.
        @parameter conversation: A List of previous conversation messages that include the role and content.

        @returns A List of message Dictionaries formatted for the chatbot. This includes an initial system message, the previous conversation messages, and the new user query encapsulated with the provided context.

        Each message in the List is a Dictionary with 'role' and 'content' keys, where 'role' is either 'system' or 'user', and 'content' contains the relevant text. This will depend on the LLM used.
        """
        messages = [
            {
                "role": "system",
                "content": "You are a Retrieval Augmented Generation chatbot. Please answer user queries only their provided context. If the provided documentation does not provide enough information, say so. If the answer requires code examples encapsulate them with ```programming-language-name ```. Don't do pseudo-code.",
            }
        ]

        for message in conversation:
            messages.append({"role": message.type, "content": message.content})

        query = " ".join(queries)
        user_context = " ".join(context)

        messages.append(
            {
                "role": "user",
                "content": f"Please answer this query: '{query}' with this provided context: {user_context}",
            }
        )

        return messages

def get_jippity_generator():
    return JippityGenerator()

available_generators.generators["jippity"] = get_jippity_generator