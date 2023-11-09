"""
Sends updates to a Discord bot.

Usage:
>>> from tqdm.contrib.discord import tqdm, trange
>>> for i in trange(10, token='{token}', channel_id='{channel_id}'):
...     ...

![screenshot](https://img.tqdm.ml/screenshot-discord.png)
"""
import logging
from os import getenv

try:
    from disco.client import Client, ClientConfig
except ImportError:
    raise ImportError("Please `pip install disco-py`")

from ..auto import tqdm as tqdm_auto
from .utils_worker import MonoWorker

__author__ = {"github.com/": ["casperdcl"]}
__all__ = ['DiscordIO', 'tqdm_discord', 'tdrange', 'tqdm', 'trange']


class DiscordIO(MonoWorker):
    """Non-blocking file-like IO using a Discord Bot."""
    def __init__(self, token, channel_id):
        """Creates a new message in the given `channel_id`."""
        super(DiscordIO, self).__init__()
        config = ClientConfig()
        config.token = token
        client = Client(config)
        self.text = self.__class__.__name__
        try:
            self.message = client.api.channels_messages_create(channel_id, self.text)
        except Exception as e:
            tqdm_auto.write(str(e))
            self.message = None

    def write(self, s):
        """Replaces internal `message`'s text with `s`."""
        if not s:
            s = "..."
        s = s.replace('\r', '').strip()
        if s == self.text:
            return  # skip duplicate message
        message = self.message
        if message is None:
            return
        self.text = s
        try:
            future = self.submit(message.edit, '`' + s + '`')
        except Exception as e:
            tqdm_auto.write(str(e))
        else:
            return future


class tqdm_discord(tqdm_auto):
    """
    Standard `tqdm.auto.tqdm` but also sends updates to a Discord Bot.
    May take a few seconds to create (`__init__`).

    - create a discord bot (not public, no requirement of OAuth2 code
      grant, only send message permissions) & invite it to a channel:
      <https://discordpy.readthedocs.io/en/latest/discord.html>
    - copy the bot `{token}` & `{channel_id}` and paste below

    >>> from tqdm.contrib.discord import tqdm, trange
    >>> for i in tqdm(iterable, token='{token}', channel_id='{channel_id}'):
    ...     ...
    """
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        token  : str, required. Discord token
            [default: ${TQDM_DISCORD_TOKEN}].
        channel_id  : int, required. Discord channel ID
            [default: ${TQDM_DISCORD_CHANNEL_ID}].
        mininterval  : float, optional.
          Minimum of [default: 1.5] to avoid rate limit.

        See `tqdm.auto.tqdm.__init__` for other parameters.
        """
        if not kwargs.get('disable'):
            kwargs = kwargs.copy()
            logging.getLogger("HTTPClient").setLevel(logging.WARNING)
            self.dio = DiscordIO(
                kwargs.pop('token', getenv("TQDM_DISCORD_TOKEN")),
                kwargs.pop('channel_id', getenv("TQDM_DISCORD_CHANNEL_ID")))
            kwargs['mininterval'] = max(1.5, kwargs.get('mininterval', 1.5))
        super(tqdm_discord, self).__init__(*args, **kwargs)

    def display(self, **kwargs):
        super(tqdm_discord, self).display(**kwargs)
        fmt = self.format_dict
        if fmt.get('bar_format', None):
            fmt['bar_format'] = fmt['bar_format'].replace(
                '<bar/>', '{bar:10u}').replace('{bar}', '{bar:10u}')
        else:
            fmt['bar_format'] = '{l_bar}{bar:10u}{r_bar}'
        self.dio.write(self.format_meter(**fmt))

    def clear(self, *args, **kwargs):
        super(tqdm_discord, self).clear(*args, **kwargs)
        if not self.disable:
            self.dio.write("")


def tdrange(*args, **kwargs):
    """Shortcut for `tqdm.contrib.discord.tqdm(range(*args), **kwargs)`."""
    return tqdm_discord(range(*args), **kwargs)


# Aliases
tqdm = tqdm_discord
trange = tdrange
