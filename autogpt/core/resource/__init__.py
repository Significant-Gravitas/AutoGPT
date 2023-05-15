from autogpt.core.resource.base import ResourceManager
from autogpt.core.resource.schema import (
    ProviderBudget,
    ProviderCredentials,
    ProviderSettings,
    ProviderUsage,
    ResourceType,
)
from autogpt.core.resource.simple import ResourceManagerSettings, SimpleResourceManager
from autogpt.core.status import ShortStatus, Status

status = Status(
    module_name=__name__,
    short_status=ShortStatus.BASIC_DONE,
    handoff_notes=(
        "Before times: Sketched out BudgetManager.__init__\n"
        "5/6: First interface draft complete.\n"
        "5/7: Basic BudgetManager has been implemented and interface adjustments made.\n"
        "5/10: BudgetManager interface revisions have been PR'ed and merged\n"
        "5/15: Pivot to make resources first class. Add many resource abstractions, port model providers,\n"
        "      port budget manager to resource manager.\n"
    ),
)
