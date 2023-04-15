from . import *
from llm_utils import create_chat_completion


def read_file(filepath, char_limit=4000):
    """Read a file and return the contents"""
    try:
        with open(filepath, "r", encoding='utf-8') as f:
            content = f.read()
            chunks = split_text(content, char_limit)
            if len(chunks) > 1:
                summaries = []

                for i, chunk in enumerate(chunks):
                    print(f"Summarizing chunk {i + 1} / {len(chunks)}")
                    messages = [create_message(chunk)]

                    summary = create_chat_completion(
                        model=cfg.fast_llm_model,
                        messages=messages,
                        max_tokens=300,
                    )
                    summaries.append(summary)

                print(f"Summarized {len(chunks)} chunks.")

                combined_summary = "\n".join(summaries)

                return combined_summary
            elif len(chunks) == 1:
                return chunks[0]
            else:
                return "Error: empty content"
    except Exception as e:
        return "Error: " + str(e)
