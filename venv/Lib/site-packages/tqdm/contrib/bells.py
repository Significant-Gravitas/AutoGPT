"""
Even more features than `tqdm.auto` (all the bells & whistles):

- `tqdm.auto`
- `tqdm.tqdm.pandas`
- `tqdm.contrib.telegram`
    + uses `${TQDM_TELEGRAM_TOKEN}` and `${TQDM_TELEGRAM_CHAT_ID}`
- `tqdm.contrib.discord`
    + uses `${TQDM_DISCORD_TOKEN}` and `${TQDM_DISCORD_CHANNEL_ID}`
"""
__all__ = ['tqdm', 'trange']
import warnings
from os import getenv

if getenv("TQDM_SLACK_TOKEN") and getenv("TQDM_SLACK_CHANNEL"):
    from .slack import tqdm, trange
elif getenv("TQDM_TELEGRAM_TOKEN") and getenv("TQDM_TELEGRAM_CHAT_ID"):
    from .telegram import tqdm, trange
elif getenv("TQDM_DISCORD_TOKEN") and getenv("TQDM_DISCORD_CHANNEL_ID"):
    from .discord import tqdm, trange
else:
    from ..auto import tqdm, trange

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    tqdm.pandas()
