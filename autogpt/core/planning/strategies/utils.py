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
    # TODO: this is a hack function for now. Trying to see what errors show up in testing.
    #   Can hopefully just replace with a call to ast.literal_eval (the function api still
    #   sometimes returns json strings with minor issues like trailing commas).
    try:
        return ast.literal_eval(json_str)
    except json.decoder.JSONDecodeError as e:
        try:
            print(f"json decode error {e}. trying literal eval")
            return ast.literal_eval(json_str)
        except Exception:
            breakpoint()
