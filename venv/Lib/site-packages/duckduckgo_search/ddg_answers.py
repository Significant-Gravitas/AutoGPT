import logging

from .utils import SESSION, _do_output

logger = logging.getLogger(__name__)


def ddg_answers(
    keywords,
    related=False,
    output=None,
):
    """DuckDuckGo instant answers. Query params: https://duckduckgo.com/params

    Args:
        keywords (str): keywords for query.
        related (bool, optional): add related topics to results. Defaults to False.
        output (Optional[str], optional): csv, json. Defaults to None.

    Returns:
        Optional[List[dict]]: DuckDuckGo instant answers results.
    """

    if not keywords:
        return None

    results = []

    # request instant answer from duckduckgo
    payload = {
        "q": f"what is {keywords}",
        "format": "json",
    }
    page_data = []
    try:
        resp = SESSION.get("https://api.duckduckgo.com/", params=payload)
        resp.raise_for_status()
        page_data = resp.json()
    except Exception:
        logger.exception("")
    if page_data:
        answer = page_data.get("AbstractText", None)
        if answer:
            results.append(
                {
                    "icon": None,
                    "text": answer,
                    "topic": None,
                    "url": page_data.get("AbstractURL", None),
                }
            )

    # request related topics from duckduckgo
    if related:
        payload = {
            "q": f"{keywords}",
            "format": "json",
        }
        page_data = []
        try:
            resp = SESSION.get("https://api.duckduckgo.com/", params=payload)
            resp.raise_for_status()
            page_data = resp.json().get("RelatedTopics", [])
        except Exception:
            logger.exception("")

        for i, row in enumerate(page_data):
            topic = row.get("Name", None)
            if not topic:
                icon = row["Icon"].get("URL", None)
                results.append(
                    {
                        "icon": f"https://duckduckgo.com{icon}" if icon else None,
                        "text": row["Text"],
                        "topic": None,
                        "url": row["FirstURL"],
                    }
                )
            else:
                for subrow in row["Topics"]:
                    icon = subrow["Icon"].get("URL", None)
                    results.append(
                        {
                            "icon": f"https://duckduckgo.com{icon}" if icon else None,
                            "text": subrow["Text"],
                            "topic": topic,
                            "url": subrow["FirstURL"],
                        }
                    )

    if output:
        _do_output("ddg_answers", keywords, output, results)

    return results
