"""
This instantiates an AutoGPT agent who is capable of handling any task.
It is designed to pass benchmarks as effectively as possible.

Loads in the ai_settings.yaml file to get the AI's name, role, and goals.
Sets the ai to continuous mode, but kills it if it takes more than 50,000 tokens on any particular evaluation.

The model is instantiated with a prompt from the AutoGPT completion function.

Eventualy we will also save and log all of the associated output and thinking for the model as well
"""
from pathlib import Path
import os


class AutoGPTAgent:
    """
    A class object that contains the configuration information for the AI
    The init function takes an evaluation prompt.
    It copies the ai_settings.yaml file in AutoGPTData to the Auto-GPT repo.
    It then copies the given prompt to a text file to Auto-GPT/auto_gpt_workspace called prompt.txt
    It then polls the token usage of the model and for a file called output.txt in the Auto-GPT/auto_gpt_workspace folder.
    If the model has used more than 50,000 tokens, it kills the model.
    If the model has used less than 50,000 tokens, it returns the output.txt file.
    """
    def _clean_up_workspace(self):
        """
        Cleans up the workspace by deleting the prompt.txt and output.txt files.
        :return:
        """
        # check if the files are there and delete them if they are
        if self.prompt_file.exists():
            self.prompt_file.unlink()
        if self.output_file.exists():
            self.output_file.unlink()

    def _copy_ai_settings(self):
        self.ai_settings_dest.write_text(self.ai_settings_file.read_text())

    def _copy_prompt(self):
        self.prompt_file.write_text(self.prompt)

    def _start_agent(self):
        """
        This starts the agent in the docker container.
        This assumes you have the docker image built with:
        docker build -t autogpt .
        In the dockerfile in the Auto-GPT repo.
        You also must set up the .env file in the Auto-GPT repo.
        :return:
        """
        env_file = self.auto_gpt_path / ".env"
        # run it in continuous mode and skip re-prompts
        os.system(f"docker run -it --env-file={env_file} -v {self.auto_workspace}:/home/appuser/auto_gpt_workspace -v {self.auto_gpt_path}/autogpt:/home/appuser/autogpt autogpt --continuous -C '/home/appuser/auto_gpt_workspace/ai_settings.yaml'")

    def _poll_for_output(self):
        """
        This polls the output file to see if the model has finished.
        :return:
        """
        while True:
            if self.output_file.exists():
                return self.output_file.read_text()

    def __init__(self, prompt):
        self.auto_gpt_path = Path(__file__).parent / "Auto-GPT"
        self.auto_workspace = self.auto_gpt_path / "auto_gpt_workspace"
        self.prompt_file = self.auto_workspace / "prompt.txt"
        self.output_file = self.auto_workspace / "output.txt"
        self.ai_settings_file = Path(__file__).parent / "AutoGPTData" / "ai_settings.yaml"
        self.ai_settings_dest = self.auto_workspace / "ai_settings.yaml"
        self.prompt = prompt
        self._clean_up_workspace()
        self._copy_ai_settings()
        self._copy_prompt()

    def start(self):
        self._start_agent()
        answer = self._poll_for_output()
        print('about to do clean up')
        print(answer)
        self._clean_up_workspace()
        print('did clean up')
        return answer




