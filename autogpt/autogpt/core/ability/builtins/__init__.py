from autogpt.core.ability.builtins.create_new_ability import CreateNewAbility
from autogpt.core.ability.builtins.query_language_model import QueryLanguageModel

BUILTIN_ABILITIES = {
    QueryLanguageModel.name(): QueryLanguageModel,
}

__all__ = [
    "BUILTIN_ABILITIES",
    "CreateNewAbility",
    "QueryLanguageModel",
]
