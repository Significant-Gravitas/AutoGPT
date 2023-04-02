import dirtyjson
from ai_functions import fix_json

def fix_and_parse_json(json_str: str, try_to_fix_with_gpt: bool = True):
    try:
        return dirtyjson.loads(json_str)
    except Exception as e:
        if try_to_fix_with_gpt:
          # Now try to fix this up using the ai_functions
          return fix_json(json_str, None, True)
        else:
          raise e