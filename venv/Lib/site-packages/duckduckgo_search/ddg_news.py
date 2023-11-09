import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from time import sleep

from .utils import SESSION, _do_output, _get_vqd, _normalize

logger = logging.getLogger(__name__)


def ddg_news(
    keywords,
    region="wt-wt",
    safesearch="moderate",
    time=None,
    max_results=None,
    page=1,
    output=None,
):
    """DuckDuckGo news search. Query params: https://duckduckgo.com/params

    Args:
        keywords (str): keywords for query.
        region (str): wt-wt, us-en, uk-en, ru-ru, etc. Defaults to "wt-wt".
        safesearch (str): on, moderate, off. Defaults to "moderate".
        time (Optional[str], optional): d, w, m. Defaults to None.
        max_results (Optional[int], optional): maximum number of results, max=240. Defaults to None.
            if max_results is set, then the parameter page is not taken into account.
        page (int, optional): page for pagination. Defaults to 1.
        output (Optional[str], optional): csv, json. Defaults to None.

    Returns:
        Optional[List[dict]]: DuckDuckGo news search results.
    """

    def get_ddg_news_page(page):
        payload["s"] = max(PAGINATION_STEP * (page - 1), 0)
        page_data = None
        try:
            resp = SESSION.get("https://duckduckgo.com/news.js", params=payload)
            resp.raise_for_status()
            page_data = resp.json().get("results", None)
        except Exception:
            logger.exception("")
            if not max_results:
                return None
        page_results = []
        if page_data:
            for row in page_data:
                if row["url"] not in cache:
                    cache.add(row["url"])
                    page_results.append(
                        {
                            "date": datetime.utcfromtimestamp(row["date"]).isoformat(),
                            "title": row["title"],
                            "body": _normalize(row["excerpt"]),
                            "url": row["url"],
                            "image": row.get("image", None),
                            "source": row["source"],
                        }
                    )
        return page_results

    if not keywords:
        return None

    # get vqd
    vqd = _get_vqd(keywords)
    if not vqd:
        return None

    PAGINATION_STEP, MAX_API_RESULTS = 30, 240

    # prepare payload
    safesearch_base = {"On": 1, "Moderate": -1, "Off": -2}
    payload = {
        "l": region,
        "o": "json",
        "noamp": "1",
        "q": keywords,
        "vqd": vqd,
        "p": safesearch_base[safesearch.capitalize()],
        "df": time,
        "s": 0,
    }

    # get results
    cache = set()
    if max_results:
        results = []
        max_results = min(abs(max_results), MAX_API_RESULTS)
        iterations = (max_results - 1) // PAGINATION_STEP + 1  # == math.ceil()
        with ThreadPoolExecutor(min(iterations, 4)) as executor:
            fs = []
            for page in range(1, iterations + 1):
                fs.append(executor.submit(get_ddg_news_page, page))
                sleep(min(iterations / 17, 0.3))  # sleep to prevent blocking
            for r in as_completed(fs):
                if r.result():
                    results.extend(r.result())
        results = results[:max_results]
    else:
        results = get_ddg_news_page(page=page)
        if not results:
            return None

    results.sort(key=lambda x: x["date"], reverse=True)

    # save to csv or json file
    if output:
        _do_output("ddg_news", keywords, output, results)

    return results
