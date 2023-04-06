import commands as cmd
import memory as mem
from colorama import Fore, Style
from spinner import Spinner
from enum import Enum, auto
from config import Config
from json_parser import fix_and_parse_json
from ai_config import AIConfig
from chat import create_chat_completion
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

def chat_with_custom_model(current_context, tokens_remaining):

    save_path=r''#Your tokenizer path
    #tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    # Convert context to input tokens
    input_context = tokenizer.encode(' '.join([msg['content'] for msg in current_context]), return_tensors='pt')

    # Check if input_context is empty or contains only special tokens
    if input_context.shape[1] == 0 or all(token in tokenizer.all_special_ids for token in input_context[0]):
        return "I'm sorry, I cannot generate a response based on an empty input."

    # Clamp max_length to model's max position embeddings
    max_length = min(len(input_context[0]) + tokens_remaining, model.config.max_position_embeddings)

    # Generate response tokens
    output_tokens = model.generate(input_context, max_length=max_length, do_sample=True, pad_token_id=tokenizer.eos_token_id)


    # Convert response tokens back to text
    assistant_reply = tokenizer.decode(output_tokens[:, input_context.shape[-1]:][0], skip_special_tokens=True)

    fixed_json = fix_and_parse_json(assistant_reply)

     # Print the fixed_json for debugging
    print("fixed_json:", fixed_json)
    return json.dumps(fixed_json)
