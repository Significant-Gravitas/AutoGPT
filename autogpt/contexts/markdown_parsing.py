import re


# Compares the header levels of the context and template to see if they match
def matches_template(context_data, template):
    context_levels = get_header_levels(context_data)
    template_levels = get_header_levels(template)
    print(f"Template Matching Test...")
    print(f"Context Levels: {context_levels}")
    print(f"Template Levels: {template_levels}")
    return context_levels == template_levels

# Breaks down the markdown string into a list of header levels to compare
def get_header_levels(markdown_string):
    if markdown_string is None:
        return []

    headers = re.findall(r'#+', markdown_string)
    return headers


# Might need this later
def extract_code_block(response: str) -> str:
    code_block_pattern = r"```(.*?)```"
    code_block_match = re.search(code_block_pattern, response, re.DOTALL)

    if code_block_match:
        return code_block_match.group(1).strip()
    else:
        return response