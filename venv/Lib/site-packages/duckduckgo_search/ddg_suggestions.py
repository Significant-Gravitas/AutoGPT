import logging

from .utils import SESSION, _do_output

logger = logging.getLogger(__name__)


def ddg_suggestions(
    keywords,
    region="wt-wt",
    output=None,
):
    """DuckDuckGo suggestions. Query params: https://duckduckgo.com/params

    Args:
        keywords (str): keywords for query.
        region (str, optional): wt-wt, us-en, uk-en, ru-ru, etc. Defaults to "wt-wt".
        output (Optional[str], optional): csv, json. Defaults to None.

    Returns:
        Optional[List[str]]: DuckDuckGo suggestions results.
    """

    if not keywords:
        return None

    results = []

    # request suggestions from duckduckgo
    payload = {
        "q": keywords,
        "kl": region,
    }
    try:
        resp = SESSION.get("https://duckduckgo.com/ac", params=payload)
        resp.raise_for_status()
        results = resp.json()
    except Exception:
        logger.exception("")

    if output:
        _do_output("ddg_suggestions", keywords, output, results)

    return results
