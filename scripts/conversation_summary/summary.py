import json
import re
import datetime
import sys
from pathlib import Path
from typing import List, Union, Callable, Any, Optional, Dict

from colorama import Style, Fore

from token_counter import count_string_tokens


class Summary:
    """
    A class to manage step-by-step and final summaries of an AI conversation.
    """

    def __init__(
            self,
            step_summarization_prompt: str,
            final_summarization_prompt: str,
            ai_name: str,
            summary_filename: str = None
    ) -> None:
        """
        Initializes the Summary class with prompts and summary file name.

        Args:
        step_summarization_prompt (str): The prompt to generate step summaries.
        final_summarization_prompt (str): The prompt to generate the final summary.
        ai_name (str): The name of the AI model.
        summary_filename (str, optional): The name of the summary file. Defaults to None.
        """

        # If a summary filename is not provided, create one using the current date and time
        if not summary_filename:
            logs_path = Path("logs")
            logs_path.mkdir(exist_ok=True)

            summary_filename = logs_path / f"{ai_name}_summary_{datetime.datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}.txt"

        self.summary_filename = summary_filename
        self.step_counter = 1
        self.step_summarization_prompt = step_summarization_prompt
        self.final_summarization_prompt = final_summarization_prompt

    def print_step_summary_to_console(self, step_summary: str) -> None:
        """
        Prints the step summary to the console.

        Args:
        step_summary (str): The step summary to print.
        """
        console_output = f"""{Fore.LIGHTYELLOW_EX}
====================================================
\nSTEP {self.step_counter} SUMMARY\n{step_summary}\n
===================================================={Style.RESET_ALL}
"""
        print(console_output)

    @staticmethod
    def print_final_summary_to_console(final_summary: str) -> None:
        """
        Prints the final summary to the console.

        Args:
        final_summary (str): The final summary to print.
        """
        console_output = f"""{Fore.LIGHTYELLOW_EX}
===================================================================
\nFINAL SUMMARY\n{final_summary}\n
==================================================================={Style.RESET_ALL}
"""
        print(console_output)

    def append_summarized_step_to_file(self, step_summary: str, step_result: str) -> None:
        """
        Appends the step summary and step result to the summary file.

        Args:
        step_summary (str): The step summary to append.
        step_result (str): The step result to append.
        """
        formatted_step_summary = self._split_sentences_into_bullet_points(step_summary)

        with open(self.summary_filename, "a", encoding='utf-8') as summary_file:
            summary_file.write("========================================\n")
            summary_file.write(f"               STEP {self.step_counter}\n")
            summary_file.write("========================================\n\n")
            summary_file.write(f"\nStep summary:\n{formatted_step_summary}\n")
            summary_file.write(f"Result: {step_result}\n")
            summary_file.write(f"\n========================================\n\n\n\n")

    def append_final_summary_to_file(
            self,
            gpt_agent_instance: Optional = None,
            create_agent_callback: Callable = None,
            next_agent_key: int = None,
            gpt_agent_model: str = "gpt-3.5-turbo",
    ) -> str:
        """
        Appends the final summary to the summary file.

        NOTE: Either provide `gpt_agent_instance` or the combination of `gpt_agent_model`, `create_agent_callback`,
        and `next_agent_key` to create a GPT agent.

        Args:
        gpt_agent_instance (Optional): The GPT agent instance. Defaults to None.
        create_agent_callback (Callable): The callback to create an agent. Defaults to None.
        next_agent_key (int): The next agent key. Defaults to None.
        gpt_agent_model (str): The GPT agent model. Defaults to "gpt-3.5-turbo".
        """
        file_content = self._read_file_content(self.summary_filename)
        print("Generating final summary...")
        final_summary = self._generate_final_summary(
            file_content=file_content,
            final_summarization_prompt=self.final_summarization_prompt,
            gpt_agent_instance=gpt_agent_instance,
            create_agent_callback=create_agent_callback,
            next_agent_key=next_agent_key,
            gpt_agent_model=gpt_agent_model
        )
        print("Final summary generated.")
        print(f"Final summary: {final_summary}")
        self._write_final_summary_to_file(
            filename=self.summary_filename,
            final_summary=final_summary,
            num_of_total_steps=self.step_counter - 1
        )

        return final_summary

    def increment_step_counter(self):
        """
        Increments the step counter.
        """
        self.step_counter += 1

    def _generate_step_summary(self) -> str:
        pass

    def _generate_final_summary(
            self,
            file_content: str,
            final_summarization_prompt: str,
            gpt_agent_instance: Optional = None,
            create_agent_callback: Callable = None,
            next_agent_key: int = None,
            gpt_agent_model: str = "gpt-3.5-turbo",
    ) -> str:
        """
        Generates the final summary based on the file content and the final summary prompt.

        NOTE: Either provide `gpt_agent_instance` or the combination of `gpt_agent_model`, `create_agent_callback`,
        and `next_agent_key` to create a GPT agent.

        Args:
        file_content (str): The content of the summary file.
        final_summarization_prompt (str): The prompt for the final summary.
        gpt_agent_instance (Optional): An instance of a GPT agent, if available.
        create_agent_callback (Callable): A callback to create a GPT agent.
        next_agent_key (int): The key for the next agent.
        gpt_agent_model (str): The name of the GPT agent model.

        Returns:
        str: The generated final summary.
        """
        text_chunks = self._split_text_into_chunks(
            text=file_content,
            # The max number of tokens is 4,097, but we need to leave some space for the prompt
            max_tokens=3980,
        )

        final_summary = ""

        for chunk in text_chunks:
            # Format the prompt with the chunk of text
            message = self._format_final_summary_prompt(final_summarization_prompt, chunk)

            # If the GPT agent instance is not provided, create one using the callback
            if not gpt_agent_instance:
                _, formatted_summary = create_agent_callback(next_agent_key, message, gpt_agent_model)
            else:
                _, formatted_summary = gpt_agent_instance.send_message(message)

            # Split the summary into bullet points
            formatted_summary = self._split_sentences_into_bullet_points(formatted_summary)

            # Append the summary to the final summary
            final_summary += f"\n{formatted_summary}"

        return final_summary

    @staticmethod
    def _format_final_summary_prompt(prompt_to_format: str, chunk: str) -> str:
        """
        Formats the final summary prompt by adding the provided chunk of text.

        Args:
        prompt_to_format (str): The prompt to format.
        chunk (str): The chunk of text to include in the prompt.

        Returns:
        str: The formatted final summary prompt.
        """
        if prompt_to_format.count("{}") != 1:
            return "The prompt string must contain one instance of '{}' which will include the chunk of text. " \
                   "Please update your FINAL_SUMMARY_PROMPT variable in the .env file."
        return prompt_to_format.format(chunk)

    @staticmethod
    def _write_final_summary_to_file(filename: str, final_summary: str, num_of_total_steps: int) -> None:
        with open(filename, "a", encoding='utf-8') as summary_file:
            summary_file.write("=======================================================\n")
            summary_file.write(f"          FINAL SUMMARY ({num_of_total_steps} TOTAL STEPS) \n")
            summary_file.write("=======================================================\n\n")
            summary_file.write(final_summary)
            summary_file.write("\n\n=======================================================\n\n")

    @staticmethod
    def _read_file_content(filename) -> str:
        if not Path(filename).exists():
            with open(filename, "w", encoding='utf-8') as summary_file:
                summary_file.write("No steps logged.\n")

        with open(filename, 'r', encoding='utf-8') as summary_file:
            file_content = summary_file.read()

        return file_content

    @staticmethod
    def _split_sentences_into_bullet_points(text: str) -> str:
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        bullet_points = "\n".join([f"- {sentence.strip()}" for sentence in sentences if sentence.strip()])
        return bullet_points

    @staticmethod
    def _split_text_into_chunks(
            text: str,
            max_tokens: int,
            gpt_agent_model: str = "gpt-3.5-turbo",
    ) -> List[str]:
        tokens = text.split()
        chunks = []
        current_chunk = []

        for token in tokens:
            current_chunk.append(token)
            current_chunk_str = " ".join(current_chunk)
            token_count = count_string_tokens(current_chunk_str, gpt_agent_model)

            if token_count > max_tokens:
                current_chunk.pop()  # Remove the last token that exceeded the limit
                chunks.append(" ".join(current_chunk))
                current_chunk = [token]  # Start a new chunk with the removed token

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


class SummaryUtils:
    """
    A utility class for working with summaries.
    """
    @staticmethod
    def get_step_summary_from_assistant_reply(
            assistant_reply: Union[str, dict],
            fix_and_parse_json: Callable[[str], Any],
            print_to_console: Callable[..., None]
    ) -> str:
        """
        Extracts the step summary from the assistant's reply.

        Args:
        assistant_reply (Union[str, dict]): The assistant's reply.
        fix_and_parse_json (Callable[[str], Any]): A function to fix and parse JSON.
        print_to_console (Callable[..., None]): A function to print to console.

        Returns:
        str: The step summary.
        """
        assistant_reply_json = fix_and_parse_json(assistant_reply)

        if isinstance(assistant_reply_json, str):
            try:
                assistant_reply_json = assistant_reply_json.replace('\n', '').replace('\r', '').replace('\t', '').replace('  ', '')
                assistant_reply_json = json.loads(assistant_reply_json)
            except json.JSONDecodeError as e:
                print_to_console("Error: Invalid JSON\n", Fore.RED, assistant_reply)
                assistant_reply_json = {}

        step_summary = assistant_reply_json.get("summary", "No summary provided.")

        return step_summary

    @staticmethod
    def add_summary_field_to_json(json_schema: Union[str, dict], value: str) -> str:
        """
        If the provided JSON schema is a dictionary, adds a summary field to it, and returns the updated JSON as a string.
        If the provided JSON schema is a string, converts it to a dictionary, adds a summary field to it, and returns the updated JSON as a string.
        """
        if isinstance(json_schema, dict):
            json_schema["summary"] = value
            return json.dumps(json_schema, sort_keys=False, indent=4)
        elif isinstance(json_schema, str):
            json_schema = json_schema.replace('\n', '').replace('\r', '').replace('\t', '').replace('  ', '')
            json_schema_dict = json.loads(json_schema)
            json_schema_dict["summary"] = value
            updated_json_schema = json.dumps(json_schema_dict, sort_keys=False, indent=4)
            return updated_json_schema
        else:
            raise ValueError("The provided JSON schema must be a dictionary or a string.")

    @staticmethod
    def add_summary_field_to_prompt(prompt: str, value: str) -> str:
        # Regular expression pattern to find the JSON object
        pattern = r'\{\s*?"thoughts":\s*?\{[\s\S]*?"args":\s*?\{[\s\S]*?\}\s*?\}\s*?\}'

        # Search for the JSON object in the prompt
        match = re.search(pattern, prompt)

        if match:
            # Extract the JSON object string
            json_string = match.group()

            # Load the JSON object as a dictionary
            json_dict = json.loads(json_string)

            # Add the "summary" field to the JSON dictionary
            json_dict["summary"] = value

            # Convert the JSON dictionary back to a string
            updated_json_string = json.dumps(json_dict, indent=4)

            # Replace the original JSON object with the updated one
            updated_prompt = prompt[:match.start()] + updated_json_string + prompt[match.end():]

            return updated_prompt

        return prompt
