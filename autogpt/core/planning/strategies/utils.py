import json


def to_numbered_list(items: list[str], no_items_response: str = "", **template_args) -> str:
    if items:
        return "\n".join(
            f"{i+1}. {item.format(**template_args)}" for i, item in enumerate(items)
        )
    else:
        return no_items_response


def json_loads(json_str: str):
    try:
        return json.loads(json_str)
    except json.decoder.JSONDecodeError as e:
        breakpoint()
