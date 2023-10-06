"""
CSV Agent

Using chat completion, query questions about a csv
"""
import os

from forge.sdk.memory.memstore_tools import add_memory

from ..forge_log import ForgeLogger
from .registry import ability
from ..llm import chat_completion_request

logger = ForgeLogger(__name__)

@ability(
    name="query_csv",
    description="Query and extract data from a CSV file using AI",
    parameters=[
        {
            "name": "query",
            "description": "detailed search query",
            "type": "string",
            "required": True,
        },
        {
            "name": "file_name",
            "description": "Name of file",
            "type": "string",
            "required": True,
        }
    ],
    output_type="str",
)
async def query_csv(agent, task_id: str, query: str, file_name: str) -> str:
    try:
        open_csv = agent.workspace.read(task_id, file_name)
        open_csv_str = open_csv.decode()

        chat = [
            {
                "role": "system",
                "content": "You are a professional Data Analyst"
            },
            {
                "role": "user",
                "content": f"CSV DATA\n{open_csv_str}"
            },
            {
                "role": "user",
                "content": f"Given this CSV data, please answer this query about it:\n{query}"
            }
        ]

        chat_completion_parms = {
            "messages": chat,
            "model": os.getenv("OPENAI_MODEL"),
            "temperature": 0.0
        }

        chat_response = await chat_completion_request(
            **chat_completion_parms
        )

        answer = chat_response["choices"][0]["message"]["content"]

        ans_mem = {"query": query, "answer": answer}
        await add_memory(
            task_id, 
            str(ans_mem), 
            "query_csv")

        return answer
    except Exception as err:
        logger.error(f"query_csv filed: {err}")
        raise err