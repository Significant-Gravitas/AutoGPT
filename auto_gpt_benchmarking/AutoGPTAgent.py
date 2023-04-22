"""
This instantiates an AutoGPT agent who is capable of handling any task.
It is designed to pass benchmarks as effectively as possible.

Loads in the ai_settings.yaml file to get the AI's name, role, and goals.
Sets the ai to continuous mode, but kills it if it takes more than 50,000 tokens on any particular evaluation.

The model is instantiated with a prompt from the AutoGPT completion function.

Eventualy we will also save and log all of the associated output and thinking for the model as well
"""
from pathlib import Path
import docker
import asyncio
import aiodocker


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
        if self.file_logger.exists():
            self.file_logger.unlink()

    def _copy_ai_settings(self) -> None:
        self.ai_settings_dest.write_text(self.ai_settings_file.read_text())

    def _copy_prompt(self) -> None:
        self.prompt_file.write_text(self.prompt)

    async def _stream_logs(self, container: aiodocker.containers.DockerContainer) -> None:
        try:
            async for line in container.log(stdout=True, stderr=True, follow=True, tail="all"):
                print(line.strip())
                await asyncio.sleep(1)
        except aiodocker.exceptions.DockerError as e:
            # Handle Docker errors (e.g., container is killed or removed)
            print('Docker error: {}'.format(e))

    async def _run_stream_logs(self) -> None:
        """
        This grabs the docker containers id and streams the logs to the console with aiodocker.
        :return: None
        """
        async with aiodocker.Docker() as docker_client:
            try:
                container = docker_client.containers.container(self.container.id)
                await self._stream_logs(container)
            except aiodocker.exceptions.DockerError as e:
                # Handle cases when the container is not found
                print('Container not found: {}'.format(e))

    def _start_agent(self):
        """
        This starts the agent in the docker container.
        This assumes you have the docker image built with:
        docker build -t autogpt .
        In the dockerfile in the Auto-GPT repo.
        You also must set up the .env file in the Auto-GPT repo.
        :return:
        """
        client = docker.from_env()
        env_file = self.auto_gpt_path / ".env"
        envs = [
            f"{line.strip()}" for line in open(
                env_file
            ) if line.strip() != "" and line.strip()[0] != "#" and line.strip()[0] != "\n"]

        self.container = client.containers.run(
            image="autogpt",
            command="--continuous -C '/home/appuser/auto_gpt_workspace/ai_settings.yaml'",
            environment=envs,
            volumes={
                self.auto_workspace: {"bind": "/home/appuser/auto_gpt_workspace", "mode": "rw"},
                f"{self.auto_gpt_path}/autogpt": {"bind": "/home/appuser/autogpt", "mode": "rw"},
            },
            stdin_open=True,
            tty=True,
            detach=True
        )
        asyncio.run(self._run_stream_logs())

    def _poll_for_output(self):
        """
        This polls the output file to see if the model has finished.
        :return:
        """
        while True:
            if self.output_file.exists():
                return self.output_file.read_text()

    def __init__(self, prompt, auto_gpt_path: str):
        self.auto_gpt_path = Path(auto_gpt_path)
        self.auto_workspace = self.auto_gpt_path / "auto_gpt_workspace"
        # if the workspace doesn't exist, create it
        if not self.auto_workspace.exists():
            self.auto_workspace.mkdir()
        self.prompt_file = self.auto_workspace / "prompt.txt"
        self.output_file = self.auto_workspace / "output.txt"
        self.file_logger = self.auto_workspace / "file_logger.txt"
        self.ai_settings_file = Path(__file__).parent / "AutoGPTData" / "ai_settings.yaml"
        self.ai_settings_dest = self.auto_workspace / "ai_settings.yaml"
        self.prompt = prompt
        self._clean_up_workspace()
        self._copy_ai_settings()
        self._copy_prompt()
        self.container = None
        self.killing = False
        self.logging_task = None

    def start(self):
        self._start_agent()
        answer = self._poll_for_output()
        print(f"Prompt was: {self.prompt}, Answer was: {answer}")
        self.kill()
        return answer

    def kill(self):
        if self.killing:
            return
        self.killing = True
        self._clean_up_workspace()
        if self.container:
            # kill the container
            try:
                self.container.kill()
                self.container.remove()
            except docker.errors.APIError:
                print('Couldn\'t find container to kill. Assuming container successfully killed itself.')
            if self.logging_task:
                self.logging_task.cancel()
        self.killing = False




