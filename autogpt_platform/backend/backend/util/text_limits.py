"""Length bounds for text a request stores.

Put the bound on the request model where the model is request-only; where a
response model inherits it, bound the write instead, or a row already longer
than the bound becomes unreadable.
"""

from typing import Annotated

from pydantic import StringConstraints

MAX_ID_LENGTH = 100
MAX_NAME_LENGTH = 100
# For a name checked at the write, where a tighter bound would refuse a
# re-save of a row already over it.
MAX_LONG_NAME_LENGTH = 500
MAX_TITLE_LENGTH = 200
MAX_CRON_LENGTH = 200
MAX_EMAIL_LENGTH = 320
MAX_SUMMARY_LENGTH = 500
MAX_DESCRIPTION_LENGTH = 2_000
MAX_URL_LENGTH = 2_048
# A token or key a caller hands us to store.
MAX_SECRET_LENGTH = 16_384
MAX_INSTRUCTIONS_LENGTH = 20_000

# How many entries a stored list field may carry.
MAX_LIST_ENTRIES = 100

IDText = Annotated[str, StringConstraints(max_length=MAX_ID_LENGTH)]
NameText = Annotated[str, StringConstraints(max_length=MAX_NAME_LENGTH)]
URLText = Annotated[str, StringConstraints(max_length=MAX_URL_LENGTH)]
SecretText = Annotated[str, StringConstraints(max_length=MAX_SECRET_LENGTH)]
