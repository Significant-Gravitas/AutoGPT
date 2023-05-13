from autogpt.core.credentials.base import CredentialsManager, ServiceCredentials
from autogpt.core.status import ShortStatus, Status

status = Status(
    module_name=__name__,
    short_status=ShortStatus.IN_PROGRESS,
    handoff_notes=(
        "5/11: Realized we need something to manage credentials now or commit ourselves to leaving a pretty big mess.\n"
        "      Created a credentials manager and a credentials provider for openai.\n"
    ),
)
