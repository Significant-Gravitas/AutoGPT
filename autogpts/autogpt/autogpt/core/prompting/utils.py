import ast
import json


def to_numbered_list(
    items: list[str], no_items_response: str = "", **template_args
) -> str:
    if items:
        return "\n".join(
            f"{i+1}. {item.format(**template_args)}" for i, item in enumerate(items)
        )
    else:
        return no_items_response


def json_loads(json_str: str):
    # TODO: this is a hack function for now. We'll see what errors show up in testing.
    #   Can hopefully just replace with a call to ast.literal_eval.
    # Can't use json.loads because the function API still sometimes returns json strings
    #   with minor issues like trailing commas.
    try:
        json_str = json_str[json_str.index("{") : json_str.rindex("}") + 1]
        return ast.literal_eval(json_str)
    except json.decoder.JSONDecodeError as e:
        try:
            print(f"json decode error {e}. trying literal eval")
            return ast.literal_eval(json_str)
        except Exception:
            breakpoint()
