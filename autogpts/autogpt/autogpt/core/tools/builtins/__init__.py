from autogpt.core.tools.builtins.create_new_ability import CreateNewAbility
from autogpt.core.tools.builtins.query_language_model import QueryLanguageModel

BUILTIN_ABILITIES = {
    QueryLanguageModel.name(): QueryLanguageModel,
}
COMMAND_CATEGORIES = [
    "autogpt.core.tools.execute_code",
    "autogpt.core.tools.file_operations",
    "autogpt.core.tools.user_interaction",
    "autogpt.core.tools.web_search",
    "autogpt.core.tools.web_selenium",
    "autogpt.core.tools.system",
    "autogpt.core.tools.image_gen",
]
