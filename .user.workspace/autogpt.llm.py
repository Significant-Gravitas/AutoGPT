# This code loads a pre-trained GPT model (in this case, the medium-sized GPT-2 model) using the AutoGPT class from the autogpt library. It then generates a response to the prompt "Hello, how are you?" using the create_chat_completion function from the autogpt.llm.chat module. The response is then printed to the console. You can also fine-tune the pre-trained models on your own custom datasets using the autogpt.training module, allowing you to create GPT models that are specifically tailored to your needs.  # noqa: E501

from autogpt import AutoGPT
from autogpt.llm.chat import create_chat_completion
from autogpt.llm.auto import AutoGPT

# Load a pre-trained GPT model
model_name = "gpt2-medium"
model = AutoGPT.from_pretrained(model_name)

# Generate a response to a prompt
prompt = "Hello, how are you?"
response = create_chat_completion(model, prompt)
print(response)
