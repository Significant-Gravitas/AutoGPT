"""
Sends updates to a Slack app.

Usage:
>>> from tqdm.contrib.slack import tqdm, trange
>>> for i in trange(10, token='{token}', channel='{channel}'):
...     ...

![screenshot](https://img.tqdm.ml/screenshot-slack.png)
"""
import logging
from os import getenv

try:
    from slack_sdk import WebClient
except ImportError:
    raise ImportError("Please `pip install slack-sdk`")

from ..auto import tqdm as tqdm_auto
from .utils_worker import MonoWorker

__author__ = {"github.com/": ["0x2b3bfa0", "casperdcl"]}
__all__ = ['SlackIO', 'tqdm_slack', 'tsrange', 'tqdm', 'trange']


class SlackIO(MonoWorker):
    """Non-blocking file-like IO using a Slack app."""
    def __init__(self, token, channel):
        """Creates a new message in the given `channel`."""
        super(SlackIO, self).__init__()
        self.client = WebClient(token=token)
        self.text = self.__class__.__name__
        try:
            self.message = self.client.chat_postMessage(channel=channel, text=self.text)
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
            future = self.submit(self.client.chat_update, channel=message['channel'],
                                 ts=message['ts'], text='`' + s + '`')
        except Exception as e:
            tqdm_auto.write(str(e))
        else:
            return future


class tqdm_slack(tqdm_auto):
    """
    Standard `tqdm.auto.tqdm` but also sends updates to a Slack app.
    May take a few seconds to create (`__init__`).

    - create a Slack app with the `chat:write` scope & invite it to a
      channel: <https://api.slack.com/authentication/basics>
    - copy the bot `{token}` & `{channel}` and paste below
    >>> from tqdm.contrib.slack import tqdm, trange
    >>> for i in tqdm(iterable, token='{token}', channel='{channel}'):
    ...     ...
    """
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        token  : str, required. Slack token
            [default: ${TQDM_SLACK_TOKEN}].
        channel  : int, required. Slack channel
            [default: ${TQDM_SLACK_CHANNEL}].
        mininterval  : float, optional.
          Minimum of [default: 1.5] to avoid rate limit.

        See `tqdm.auto.tqdm.__init__` for other parameters.
        """
        if not kwargs.get('disable'):
            kwargs = kwargs.copy()
            logging.getLogger("HTTPClient").setLevel(logging.WARNING)
            self.sio = SlackIO(
                kwargs.pop('token', getenv("TQDM_SLACK_TOKEN")),
                kwargs.pop('channel', getenv("TQDM_SLACK_CHANNEL")))
            kwargs['mininterval'] = max(1.5, kwargs.get('mininterval', 1.5))
        super(tqdm_slack, self).__init__(*args, **kwargs)

    def display(self, **kwargs):
        super(tqdm_slack, self).display(**kwargs)
        fmt = self.format_dict
        if fmt.get('bar_format', None):
            fmt['bar_format'] = fmt['bar_format'].replace(
                '<bar/>', '`{bar:10}`').replace('{bar}', '`{bar:10u}`')
        else:
            fmt['bar_format'] = '{l_bar}`{bar:10}`{r_bar}'
        if fmt['ascii'] is False:
            fmt['ascii'] = [":black_square:", ":small_blue_diamond:", ":large_blue_diamond:",
                            ":large_blue_square:"]
            fmt['ncols'] = 336
        self.sio.write(self.format_meter(**fmt))

    def clear(self, *args, **kwargs):
        super(tqdm_slack, self).clear(*args, **kwargs)
        if not self.disable:
            self.sio.write("")


def tsrange(*args, **kwargs):
    """Shortcut for `tqdm.contrib.slack.tqdm(range(*args), **kwargs)`."""
    return tqdm_slack(range(*args), **kwargs)


# Aliases
tqdm = tqdm_slack
trange = tsrange
