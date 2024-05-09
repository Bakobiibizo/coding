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
            context=[]
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
            attention_mask = input.get("attention_mask", None)  # Get the attention mask, if provided
            input_len = input["input_ids"].shape[1]
            msg.info(f"Tokenized finished with {input_len} tokens")

            position_ids = torch.arange(0, input_len, dtype=torch.long, device=self.device).unsqueeze(0)

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
                current_token = self.tokenizer.convert_ids_to_tokens([current_token_id], skip_special_tokens=False)
                if type(current_token) == List:
                    current_token = " ".join(current_token)
                current_token = current_token.replace("<0x0A>", "\n").replace("▁", " ")
                # Update input for next iteration
                input["input_ids"] = torch.cat((input["input_ids"], current_token_id.view(1, 1)), dim=1)
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

        llama_prompt += f"Answer this query: '{query}' with this context: {user_context} [/INST] "

        return llama_prompt

def get_huggingface_generator(api_key: str):
    return HuggingFaceGenerator(api_key=api_key)

available_generators.generators["huggingface"] = get_huggingface_generator