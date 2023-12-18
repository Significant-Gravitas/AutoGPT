import ast
import json
import re

from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)


def to_numbered_list(
    items: list[str], no_items_response: str = "", **template_args
) -> str:
    if items:
        return "\n".join(
            f"{i+1}. {item.format(**template_args)}" for i, item in enumerate(items)
        )
    else:
        return no_items_response


def indent(content: str, indentation: int | str = 4) -> str:
    if type(indentation) == int:
        indentation = " " * indentation
    return indentation + content.replace("\n", f"\n{indentation}")


def to_dotted_list(
    items: list[str], no_items_response: str = "", **template_args
) -> str:
    if items:
        return "\n".join(
            f" - {item.format(**template_args)}" for i, item in enumerate(items)
        )
    else:
        return no_items_response


def to_string_list(string_list) -> str:
    if not string_list:
        raise ValueError("Input list cannot be empty")

    formatted_string = ", ".join(string_list[:-1]) + ", and " + string_list[-1]
    return formatted_string


def to_md_quotation(text):
    """
    Transforms a given string into a Markdown blockquote.

    Parameters:
    text (str): The string to be transformed.

    Returns:
    str: The transformed string as a Markdown blockquote.
    """
    # Split the text into lines
    lines = text.split("\n")

    # Prefix each line with "> "
    quoted_lines = [f"> {line}" for line in lines]

    # Join the lines back into a single string
    quoted_text = "\n".join(quoted_lines)

    return quoted_text


def json_loads(json_str: str):
    # TODO: this is a hack function for now. Trying to see what errors show up in testing.
    #   Can hopefully just replace with a call to ast.literal_eval (the function api still
    #   sometimes returns json strings with minor issues like trailing commas).

    try:
        json_str = json_str[json_str.index("{") : json_str.rindex("}") + 1]
        return ast.literal_eval(json_str)
    except Exception as e:
        LOG.warning(f"First attempt failed: {e}. Trying JSON.loads()")
    try:
        return json.loads(json_str)
    except Exception as e:
        try:
            LOG.warning(f"JSON decode error {e}. trying literal eval")

            def replacer(match):
                # Escape newlines in the matched value
                return match.group(0).replace("\n", "\\n").replace("\t", "\\t")

            # Find string values and apply the replacer function to each
            json_str = re.sub(r'".+?"', replacer, json_str)
            return ast.literal_eval(json_str)

            # NOTE: BACKUP PLAN :
            # json_str = escape_backslaches_in_json_values(json_str) # DOUBLE BACKSLASHES
            # return_json_value = ast.literal_eval(json_str)
            # return remove_double_ backslaches(return_json_value) # CONVERT DOUBLE
        except Exception:
            breakpoint()
