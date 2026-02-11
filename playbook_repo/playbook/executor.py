from .brain_client import BrainClient
from .utils import get_logger

logger = get_logger("executor")

class Executor:
    def __init__(self, brain_client: BrainClient):
        self.brain = brain_client

    async def execute_files(self, session_id: str, files: list):
        """
        Executes the trained playbook on a batch of files.
        files: list of dicts with 'filename' and 'content_b64'
        """
        logger.info(f"Executing session {session_id} on {len(files)} files.")
        return await self.brain.execute(session_id, files)
