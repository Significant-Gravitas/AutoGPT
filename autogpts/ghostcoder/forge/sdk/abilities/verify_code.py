from forge.sdk.abilities.registry import ability
from ghostcoder import FileRepository
from ghostcoder.actions.verify.code_verifier import CodeVerifier

#@ability(
#    name="verify_code",
#    description="Use this to run all tests after you updated code or tests.",
#    parameters=[
#        {
#            "name": "language",
#            "description": "Programming language of the code to be verified.",
#            "type": "string",
#            "required": False,
#        }],
#    output_type="None",
#)
async def verify_code(
    agent,
    task_id: str,
    language: str = "python"
) -> str:
    repo_dir = agent.workspace.base_path / task_id

    repository = FileRepository(repo_path=repo_dir, use_git=False)
    code_verifier = CodeVerifier(repository=repository, language="python")
    message = code_verifier.execute()

    return message.to_prompt()
