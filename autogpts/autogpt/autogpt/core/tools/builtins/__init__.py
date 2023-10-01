from autogpt.core.tools.builtins.create_new_tool import CreateNewTool
from autogpt.core.tools.builtins.query_language_model import QueryLanguageModel

BUILTIN_TOOLS = {
    QueryLanguageModel.name(): QueryLanguageModel,
}
COMMAND_CATEGORIES = [
    "autogpt.core.tools.builtins.execute_code",
    "autogpt.core.tools.builtins.file_operations",
    "autogpt.core.tools.builtins.user_interaction",
    "autogpt.core.tools.builtins.web_search",
    "autogpt.core.tools.builtins.web_selenium",
    "autogpt.core.tools.builtins.system",
    "autogpt.core.tools.builtins.image_gen",
]
