from autogpt.core.ability.builtins.query_language_model import QueryLanguageModel
from autogpt.core.ability.builtins.create_new_ability import CreateNewAbility

BUILTIN_ABILITIES = {
    QueryLanguageModel.name(): QueryLanguageModel,
}
