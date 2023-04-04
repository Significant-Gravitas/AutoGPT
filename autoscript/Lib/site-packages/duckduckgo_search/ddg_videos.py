import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep

from .utils import SESSION, _do_output, _get_vqd

logger = logging.getLogger(__name__)


def ddg_videos(
    keywords,
    region="wt-wt",
    safesearch="moderate",
    time=None,
    resolution=None,
    duration=None,
    license_videos=None,
    max_results=None,
    page=1,
    output=None,
):
    """DuckDuckGo videos search. Query params: https://duckduckgo.com/params

    Args:
        keywords (str): keywords for query.
        region (str, optional): wt-wt, us-en, uk-en, ru-ru, etc. Defaults to "wt-wt".
        safesearch (str, optional): on, moderate, off. Defaults to "moderate".
        time (Optional[str], optional): d, w, m. Defaults to None.
        resolution (Optional[str], optional): high, standart. Defaults to None.
        duration (Optional[str], optional): short, medium, long. Defaults to None.
        license_videos (Optional[str], optional): creativeCommon, youtube. Defaults to None.
        max_results (Optional[int], optional): maximum number of results, max=1000. Defaults to None.
            if max_results is set, then the parameter page is not taken into account.
        page (int, optional): page for pagination. Defaults to 1.
        output (Optional[str], optional): csv, json. Defaults to None.

    Returns:
        Optional[List[dict]]: DuckDuckGo videos search results
    """

    def get_ddg_videos_page(page):
        payload["s"] = max(PAGINATION_STEP * (page - 1), 0)
        page_data = None
        try:
            resp = SESSION.get("https://duckduckgo.com/v.js", params=payload)
            resp.raise_for_status()
            page_data = resp.json().get("results", None)
        except Exception:
            logger.exception("")
            if not max_results:
                return None
        page_results = []
        if page_data:
            for row in page_data:
                if row["content"] not in cache:
                    page_results.append(row)
                    cache.add(row["content"])
        return page_results

    if not keywords:
        return None

    # get vqd
    vqd = _get_vqd(keywords)
    if not vqd:
        return None

    PAGINATION_STEP, MAX_API_RESULTS = 60, 1000

    # prepare payload
    safesearch_base = {"On": 1, "Moderate": -1, "Off": -2}
    time = f"publishedAfter:{time}" if time else ""
    resolution = f"videoDefinition:{resolution}" if resolution else ""
    duration = f"videoDuration:{duration}" if duration else ""
    license_videos = f"videoLicense:{license_videos}" if license_videos else ""
    payload = {
        "l": region,
        "o": "json",
        "s": 0,
        "q": keywords,
        "vqd": vqd,
        "f": f"{time},{resolution},{duration},{license_videos}",
        "p": safesearch_base[safesearch.capitalize()],
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
                fs.append(executor.submit(get_ddg_videos_page, page))
                sleep(min(iterations / 17, 0.3))  # sleep to prevent blocking
            for r in as_completed(fs):
                if r.result():
                    results.extend(r.result())
        results = results[:max_results]
    else:
        results = get_ddg_videos_page(page=page)
        if not results:
            return None

    if output:
        # save to csv or json file
        _do_output("ddg_videos", keywords, output, results)

    return results
