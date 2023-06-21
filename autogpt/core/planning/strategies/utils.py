import json


def to_numbered_list(items: list[str], **template_args) -> str:
    return "\n".join(
        f"{i+1}. {item.format(**template_args)}" for i, item in enumerate(items)
    )


def json_loads(json_str: str):
    try:
        return json.loads(json_str)
    except json.decoder.JSONDecodeError as e:
        breakpoint()