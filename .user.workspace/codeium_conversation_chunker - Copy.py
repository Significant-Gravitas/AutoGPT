import re
import pyperclip

chunk_size = 50

with open(r'C:\X-hub\Gravitas-Significant\OptimalPrime-GPT\.vscode\.user.workspace\codeium_conversation_summary.txt', 'r') as file:  # noqa: E501
    data = file.readlines()

chunks = []
chunk = ""

with open('output.txt', 'w') as output_file:
    for line in data:
        try:
            if "📋 Copy code" in line:
                if len(chunk.split("\n")) > 0:
                    chunks.append(chunk.strip())
                    output_file.write(chunk + "\n")
                    output_file.write("-" * 80 + "\n")
                    chunk = ""
                chunks.append(line.strip())
                output_file.write(line)
                output_file.write("-" * 80 + "\n")
            elif re.match(r'^\w{3}, \w{3} \d{1,2}, \d{4}', line):
                if len(chunk.split("\n")) >= chunk_size:
                    chunks.append(chunk.strip())
                    output_file.write(chunk + "\n")
                    output_file.write("-" * 80 + "\n")
                    chunk = ""
                chunk += line
            else:
                chunk += line

        except Exception as e:
            print(f"Error processing line {line}: {e}")

    if len(chunk.split("\n")) > 0:
        chunks.append(chunk.strip())
        output_file.write(chunk + "\n")
        output_file.write("-" * 80 + "\n")

for chunk in chunks:
    try:
        if "📋 Copy code" in chunk:
            # process code copy chunk here
            print("Code copy chunk:")
        else:
            # process regular chunk here
            print("Regular chunk:")
        print(chunk)
    except Exception as e:
        print(f"Error processing chunk {chunk}: {e}")

output = "\n".join(chunks)
print(f"Processed {len(chunks)} chunks. Output:\n{output}")

# copy output to clipboard cache
pyperclip.copy(output)
print("Output copied to clipboard cache.")
