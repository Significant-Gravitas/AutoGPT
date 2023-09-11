from pathlib import Path

from fastapi import FastAPI
from fastapi import (
    HTTPException as FastAPIHTTPException,  # Import HTTPException from FastAPI
)
from fastapi.responses import FileResponse

app = FastAPI()


@app.get("/skill_tree")
def get_skill_tree() -> dict:
    return {
        "graph": {
            "nodes": {
                "TestWriteFile": {
                    "name": "TestWriteFile",
                    "input": "Write the word 'Washington' to a .txt file",
                    "task_id": "fde559f8-3ab8-11ee-be56-0242ac120002",
                    "category": ["interface"],
                    "dependencies": [],
                    "cutoff": 60,
                    "ground": {
                        "answer": "The word 'Washington', printed to a .txt file named anything",
                        "should_contain": ["Washington"],
                        "should_not_contain": [],
                        "files": [".txt"],
                        "eval": {"type": "file"},
                    },
                    "info": {
                        "difficulty": "interface",
                        "description": "Tests the agents ability to write to a file",
                        "side_effects": [""],
                    },
                },
                "TestReadFile": {
                    "name": "TestReadFile",
                    "category": ["interface"],
                    "task_id": "fde559f8-3ab8-11ee-be56-0242ac120002",
                    "input": "Read the file called file_to_read.txt and write its content to a file called output.txt",
                    "dependencies": ["TestWriteFile"],
                    "cutoff": 60,
                    "ground": {
                        "answer": "The content of output.txt should be 'Hello World!'",
                        "should_contain": ["Hello World!"],
                        "files": ["output.txt"],
                        "eval": {"type": "file"},
                    },
                    "info": {
                        "description": "Tests the ability for an agent to read a file.",
                        "difficulty": "interface",
                        "side_effects": [""],
                    },
                    "artifacts": [
                        {
                            "artifact_id": "a1b259f8-3ab8-11ee-be56-0242ac121234",
                            "file_name": "file_to_read.txt",
                            "file_path": "interface/write_file/artifacts_out",
                        }
                    ],
                },
            },
            "edges": [{"source": "TestWriteFile", "target": "TestReadFile"}],
        }
    }


@app.get("/agent/tasks/{challenge_id}/artifacts/{artifact_id}")
def get_artifact(
    challenge_id: str, artifact_id: str
) -> FileResponse:  # Added return type annotation
    try:
        # Look up the file path using the challenge ID and artifact ID

        file_path = "challenges/interface/read_file/artifacts_in/file_to_read.txt"
        current_directory = Path(__file__).resolve().parent

        # Return the file as a response
        return FileResponse(current_directory / file_path)

    except KeyError:
        raise FastAPIHTTPException(status_code=404, detail="Artifact not found")
