import os
import logging
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from autogpt.config import Config
from autogpt.commands.command import command
from autogpt.logs import logger
import traceback

load_dotenv()


class CodeGenerator:
    def __init__(self, checkpoint="bigcode/tiny_starcoder_py", device="cuda"):
        self.checkpoint = checkpoint
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint).to(self.device)

        self.prompt_methods = {
            "Function Signature": self._generate_function_signature,
            "Comment": self._generate_comment,
            "Docstring": self._generate_docstring,
            "Fill in the Middle": self._generate_fill_in_the_middle,
        }

    def generate(
        self,
        input_text: str,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 100,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0
    ) -> str:
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)

        try:
            outputs = self.model.generate(
                inputs,
                pad_token_id=self.tokenizer.eos_token_id,
                max_length=max_new_tokens + inputs.shape[-1],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            return self.tokenizer.decode(outputs[0])
        except Exception as e:
            error_message = f"Code generation failed with error: {str(e)}"
            traceback.print_exc()  # Print the full traceback for debugging
            raise ValueError(error_message) from e

    def _generate_fill_in_the_middle(self, input_text: str) -> str:
        return self.generate(input_text, max_new_tokens=8000)

    def _generate_function_signature(self, signature: str) -> str:
        try:
            return self.generate(f"def {signature}:", max_new_tokens=8000)
        except Exception as e:
            error_message = f"Failed to generate code with function signature '{signature}': {str(e)}"
            traceback.print_exc()  # Print the full traceback for debugging
            raise ValueError(error_message) from e

    def _generate_comment(self, comment: str) -> str:
        try:
            return self.generate(f"# {comment}\n", max_new_tokens=8000)
        except Exception as e:
            error_message = f"Failed to generate code with comment '{comment}': {str(e)}"
            traceback.print_exc()  # Print the full traceback for debugging
            raise ValueError(error_message) from e

    def _generate_docstring(self, docstring: str) -> str:
        try:
            return self.generate(f"\"\"\" {docstring} \"\"\"\n", max_new_tokens=8000)
        except Exception as e:
            error_message = f"Failed to generate code with docstring '{docstring}': {str(e)}"
            traceback.print_exc()  # Print the full traceback for debugging
            raise ValueError(error_message) from e


code_generator = CodeGenerator()

@command(
    "generate_code_signature",
    "Generate Code From Function Signature",
    args='"input_text": str',
)
def generate_code_signature(input_text: str, config: 'Config') -> str:
    """Generate code using a function signature as the input text."""
    return code_generator._generate_function_signature(input_text)


@command(
    "generate_code_comment",
    "Generate Code From Comment",
    args='"input_text": str',
)
def generate_code_comment(input_text: str, config: 'Config') -> str:
    """Generate code using a comment as the input text."""
    return code_generator._generate_comment(input_text)


@command(
    "generate_code_docstring",
    "Generate Code From Docstring",
    args='"input_text": str',
)
def generate_code_docstring(input_text: str, config: 'Config') -> str:
    """Generate code using a docstring as the input text."""
    return code_generator._generate_docstring(input_text)


@command(
    "generate_code_fill_in",
    "Generate Code From Fill In The Middle Prompt",
    args='"input_text": str',
)
def generate_code_fill_in(input_text: str, config: 'Config') -> str:
    """Generate code using a 'fill in the middle' input text."""
    return code_generator._generate_fill_in_the_middle(input_text)

