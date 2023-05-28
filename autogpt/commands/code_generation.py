import logging
import os
from dotenv import load_dotenv
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from autogpt.config import Config
from autogpt.commands.command import command
from autogpt.logs import logger

load_dotenv()

class CodeGenerationConfig:
    def __init__(
        self,
        max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", 128)),
        temperature: float = float(os.getenv("TEMPERATURE", 0.2)),
        top_k: int = int(os.getenv("TOP_K", 50)),
        top_p: float = float(os.getenv("TOP_P", 0.1)),
        repetition_penalty: float = float(os.getenv("REPETITION_PENALTY", 1.17))
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

@command(
    "generate_code",
    "Generate Code",
    '"input_text": "<input_text>"',
    lambda config: True,
    "",
)
def generate_code(input_text: str, config: 'Config') -> str:
    """Generate code using the input text.

    Args:
        input_text (str): The input text for code generation.
        config (Config): The configuration object.

    Returns:
        str: The generated code.
    """
    checkpoint = os.getenv("CHECKPOINT", "bigcode/tiny_starcoder_py")
    device = os.getenv("DEVICE", "cuda")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    code_config = CodeGenerationConfig()
    logger = logging.getLogger("CodeGenerator")

    def generate(
        input_text: str,
        config: Optional[CodeGenerationConfig] = None
    ) -> str:
        config = config or code_config
        inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
        params = {
            'max_new_tokens': config.max_new_tokens,
            'temperature': config.temperature,
            'top_k': config.top_k,
            'top_p': config.top_p,
            'repetition_penalty': config.repetition_penalty
        }
        try:
            outputs = model.generate(inputs, pad_token_id=tokenizer.eos_token_id, **params)
            return tokenizer.decode(outputs[0])
        except Exception as e:
            logger.error(f"Code generation failed with error: {str(e)}")
            raise

    def function_signature(signature: str) -> str:
        if not isinstance(signature, str):
            raise ValueError("Signature must be a string.")
        if not signature.isidentifier():
            raise ValueError("Invalid function signature. Signature must be a valid identifier.")
        try:
            return generate(f"def {signature}:")
        except Exception as e:
            logger.error(f"Failed to generate code with function signature '{signature}': {str(e)}")
            raise

    def comment_prompt(comment: str) -> str:
        if not isinstance(comment, str):
            raise ValueError("Comment must be a string.")
        try:
            return generate(f"# {comment}\n")
        except Exception as e:
            logger.error(f"Failed to generate code with comment '{comment}': {str(e)}")
            raise

    def docstring_prompt(docstring: str) -> str:
        if not isinstance(docstring, str):
            raise ValueError("Docstring must be a string.")
        try:
            return generate(f"\"\"\" {docstring} \"\"\"\n")
        except Exception as e:
            logger.error(f"Failed to generate code with docstring '{docstring}': {str(e)}")
            raise

    def fill_in_middle(input_text: str) -> str:
        if not isinstance(input_text, str):
            raise ValueError("Input text must be a string.")
        try:
            return generate(input_text)
        except Exception as e:
            logger.error(f"Failed to generate code with input text '{input_text}': {str(e)}")
            raise

    # Prompt Style 1: Function Signature
    function_signature_code = function_signature("print_hello_world")
    logger.info(f"Function Signature Code:\n{function_signature_code}")

    # Prompt Style 2: A Comment
    comment_code = comment_prompt("a python function that says hello")
    logger.info(f"Comment Code:\n{comment_code}")

    # Prompt Style 3: A Docstring
    docstring_code = docstring_prompt("a python function that says hello")
    logger.info(f"Docstring Code:\n{docstring_code}")

    # Prompt Style 4: Fill in the Middle
    input_text = "<fim_prefix>def print_one_two_three():\n    print('one')\n    <fim_suffix>\n    print('three')<fim_middle>"
    fill_in_middle_code = fill_in_middle(input_text)
    logger.info(f"Fill in the Middle Code:\n{fill_in_middle_code}")

    return ""  # Placeholder, replace with appropriate return statement

