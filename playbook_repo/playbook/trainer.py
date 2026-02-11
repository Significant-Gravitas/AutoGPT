from .brain_client import BrainClient
from .utils import get_logger

logger = get_logger("trainer")

class Trainer:
    def __init__(self, brain_client: BrainClient):
        self.brain = brain_client

    async def submit_mapping(self, session_id: str, mapping: list):
        """
        Submits the mapping to the Brain for training.
        """
        logger.info(f"Submitting mapping for session {session_id} with {len(mapping)} rules.")
        return await self.brain.train(session_id, mapping)

    async def finish(self, session_id: str):
        """
        Finalizes the training session.
        """
        logger.info(f"Finishing training for session {session_id}.")
        return await self.brain.finish_training(session_id)
