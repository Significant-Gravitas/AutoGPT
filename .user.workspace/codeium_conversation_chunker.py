import re
import os

# get absolute path of script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# construct absolute file paths
summary_file = os.path.join(script_dir, "codeium_conversation_summary.txt")
code_file = os.path.join(script_dir, "codeium_code.txt")
output_file = os.path.join(script_dir, "codeium_conversation_output.txt")

# read summary file
with open(summary_file, "r") as file:
    summary = file.read()

# ...


def print_output(output_str):
    print("-" * 80)
    print(output_str)
    print("-" * 80)


try:
    # find code blocks in summary file
    code_blocks = re.findall(r"(?:📋 )?Copy code(.+?)^#", summary, flags=re.M | re.S)

    print(f"Number of code blocks found: {len(code_blocks)}")
    for block in code_blocks:
        print("Code block:")
        print(block)

    # write code blocks to codeium_code.txt file
    with open(code_file, "w") as file:
        for block in code_blocks:
            file.write(block.strip() + "\n")
            file.write("-" * 80 + "\n")

    # remove code blocks from the summary file
    for block in code_blocks:
        summary = summary.replace(block, "")

    chunk_size = 50
    regular_chunks = []
    conversation_chunks = []
    conversation_chunk = ""

    # extract regular conversation chunks and remove empty lines
    for line in summary.split("\n"):
        try:
            if re.search(r"(?:📋 )?Copy code", line):
                # found a code block, so end the current conversation chunk
                if len(conversation_chunk.split("\n")) > 0:
                    conversation_chunks.append(conversation_chunk.strip())
                    conversation_chunk = ""
            elif re.match(r"^\w{3}, \w{3} \d{1,2}, \d{4}, \d{1,2}:\d{2} [ap]m", line):
                # found a timestamp, so end the current conversation chunk if it is too big
                if len(conversation_chunk.split("\n")) >= chunk_size:
                    conversation_chunks.append(conversation_chunk.strip())
                    conversation_chunk = ""
                conversation_chunk += line + "\n"

                if match := re.search(
                    r"(?:(?!BA|avatar).)*?\n(.*?)(?=\n\w{3}, \w{3} \d{1,2}, \d{4}, \d{1,2}:\d{2} [ap]m|$)",
                    summary[len(line) :],
                    flags=re.S,
                ):
                    conversation_chunk += match[1] + "\n"
            else:
                # regular line, so add it to the current conversation chunk
                conversation_chunk += line + "\n"
        except Exception as e:
            print(f"Error processing line: {line.strip()}, {e}")

    if len(conversation_chunk.split("\n")) > 0:
        conversation_chunks.append(conversation_chunk.strip())

    # write conversation chunks to codeium_conversation_output.txt file
    with open(output_file, "w") as file:
        for chunk in conversation_chunks:
            file.write(chunk.strip() + "\n\n")

    # overwrite regular conversation chunks in summary file
    with open(summary_file, "w") as file:
        file.write("Codeium Conversation Summary\n")
        file.write("=" * 80 + "\n\n")
        file.write("\n".join(conversation_chunks) + "\n")

    print_output("Script completed successfully")
except Exception:
    print_output
