from forge.sdk.abilities.registry import ability
from ghostcoder import FileRepository
from ghostcoder.actions.verify.code_verifier import CodeVerifier
from ghostcoder.test_tools.verify_python_pytest import PythonPytestTestTool


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
    test_tool = PythonPytestTestTool(current_dir=repo_dir, test_file_pattern="*.py")
    verifier = CodeVerifier(repository=repository, test_tool=test_tool)
    message = verifier.execute()

    return message.to_prompt()
