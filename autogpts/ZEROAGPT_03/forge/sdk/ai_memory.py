"""
Load documents in AI and for QnA
"""
import os

from .forge_log import ForgeLogger
from .memory.memstore import ChromaMemStore
from . import Workspace
from . import chat_completion_request

logger = ForgeLogger(__name__)

class AIMemory:
    """
    Takes in query, finds relevant document in memstore
    then creates a prompt to query the document with query also

    Still limited on long data
    """
    def __init__(
        self,
        workspace: Workspace,
        task_id: str,
        query: str,
        question: str = None,
        model: str = os.getenv("OPENAI_MODEL")):

        self.workspace = workspace
        self.task_id = task_id
        self.query = query
        self.model = model
        self.question = question if question else query

        self.chat = []
        self.relevant_doc = None
        self.prompt = None

    def get_doc(self) -> None:
        """
        Get document from VecStor
        """
        try:
            # find doc in chromadb
            cwd = self.workspace.get_cwd_path(self.task_id)
            chroma_dir = f"{cwd}/chromadb/"

            memory = ChromaMemStore(chroma_dir)
            memory_resp = memory.query(
                task_id=self.task_id,
                query=self.query
            )

            self.relevant_doc = memory_resp["documents"][0][0]
        except Exception as err:
            logger.error(f"get_doc failed: {err}")
            raise err
    
    async def query_doc_ai(self) -> str:
        """
        Uses doc found from VecStor and creates a QnA agent
        """
        if self.relevant_doc:
            self.prompt = f"""
            You are 'The Librarian' a bot that answers questions using text from the reference document included below.
            If the passage is irrelevant to the answer, you may ignore it.
            DOCUMENT: '{self.relevant_doc}'
            """

            self.chat.append({
                "role": "system",
                "content": self.prompt
            })

            self.chat.append({
                "role": "user",
                "content": f"{self.question}"
            })

            logger.info(f"Sending Doc QnA Chat\n{self.chat}")

            try:
                chat_completion_parms = {
                    "messages": self.chat,
                    "model": self.model,
                    "temperature": 0
                }

                response = await chat_completion_request(
                    **chat_completion_parms)
                
                resp_content = response["choices"][0]["message"]["content"]

                logger.info(f"reponse: {resp_content}")

                return resp_content
            except Exception as err:
                logger.error(f"chat completion failed: {err}")
                return "chat completion failed, document might be too large"
        else:
            logger.error("no relevant_doc found")
            return "no relevant document found"


