import dirtyjson
from call_ai_function import call_ai_function
from config import Config
cfg = Config()

def fix_and_parse_json(json_str: str, try_to_fix_with_gpt: bool = True):
    try:
        return dirtyjson.loads(json_str)
    except Exception as e:
        # Let's do something manually - sometimes GPT responds with something BEFORE the braces:
        # "I'm sorry, I don't understand. Please try again."{"text": "I'm sorry, I don't understand. Please try again.", "confidence": 0.0}
        # So let's try to find the first brace and then parse the rest of the string
        try:
          brace_index = json_str.index("{")
          json_str = json_str[brace_index:]
          last_brace_index = json_str.rindex("}")
          json_str = json_str[:last_brace_index+1]
          return dirtyjson.loads(json_str)
        except Exception as e:
          if try_to_fix_with_gpt:
            # Now try to fix this up using the ai_functions
            return fix_json(json_str, None, True)
          else:
            raise e
        
# TODO: Make debug a global config var
def fix_json(json_str: str, schema:str = None, debug=True) -> str:
    # Try to fix the JSON using gpt:
    function_string = "def fix_json(json_str: str, schema:str=None) -> str:"
    args = [json_str, schema]
    description_string = """Fixes the provided JSON string to make it parseable. If the schema is provided, the JSON will be made to look like the schema, otherwise it will be made to look like a valid JSON object."""

    # If it doesn't already start with a "`", add one:
    if not json_str.startswith("`"):
      json_str = "```json\n" + json_str + "\n```"
    result_string = call_ai_function(
        function_string, args, description_string, model=cfg.fast_llm_model
    )
    if debug:
        print("------------ JSON FIX ATTEMPT ---------------")
        print(f"Original JSON: {json_str}")
        print(f"Fixed JSON: {result_string}")
        print("----------- END OF FIX ATTEMPT ----------------")
    try:
        return dirtyjson.loads(result_string)
    except:
        # Log the exception:
        print("Failed to fix JSON")
        # Get the call stack:
        import traceback
        call_stack = traceback.format_exc()
        print(call_stack)
        return {}
