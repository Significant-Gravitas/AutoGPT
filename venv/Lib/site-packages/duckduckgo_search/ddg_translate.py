import logging

from .utils import SESSION, VQD_DICT, _do_output, _get_vqd

logger = logging.getLogger(__name__)


def ddg_translate(
    keywords,
    from_=None,
    to="en",
    output=None,
):
    """DuckDuckGo translate

    Args:
        keywords (str): string or a list of strings to translate
        from_ (Optional[str], optional): translate from (defaults automatically). Defaults to None.
        to (str): what language to translate. Defaults to "en".
        output (Optional[str], optional): csv, json. Defaults to None.

    Returns:
        Optional[List[dict]]: DuckDuckGo translate results.
    """

    if not keywords:
        return None

    # get vqd
    vqd = _get_vqd("translate")
    if not vqd:
        return None

    # translate
    payload = {
        "vqd": vqd,
        "query": "translate",
        "from": from_,
        "to": to,
    }

    if isinstance(keywords, str):
        keywords = [keywords]

    results = []
    for data in keywords:
        try:
            resp = SESSION.post(
                "https://duckduckgo.com/translation.js",
                params=payload,
                data=data.encode("utf-8"),
            )
            resp.raise_for_status()
            result = resp.json()
            result["original"] = data
            results.append(result)
        except Exception:
            VQD_DICT.pop("translate", None)
            logger.exception("")

    if output:
        keywords = keywords[0]
        _do_output("ddg_translate", keywords, output, results)
    return results
