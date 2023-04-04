"""Duckduckgo_search
~~~~~~~~~~~~~~
Search for words, documents, images, videos, news, maps and text translation
using the DuckDuckGo.com search engine.
"""

import logging

from .ddg import ddg
from .ddg_answers import ddg_answers
from .ddg_images import ddg_images
from .ddg_maps import ddg_maps
from .ddg_news import ddg_news
from .ddg_suggestions import ddg_suggestions
from .ddg_translate import ddg_translate
from .ddg_videos import ddg_videos
from .version import __version__

# A do-nothing logging handler
# https://docs.python.org/3.3/howto/logging.html#configuring-logging-for-a-library
logging.getLogger("duckduckgo_search").addHandler(logging.NullHandler())
