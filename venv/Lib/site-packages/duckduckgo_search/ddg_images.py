import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from time import sleep
from unicodedata import normalize

from click import progressbar

from .utils import SESSION, _do_output, _download_file, _get_vqd

logger = logging.getLogger(__name__)


def ddg_images(
    keywords,
    region="wt-wt",
    safesearch="moderate",
    time=None,
    size=None,
    color=None,
    type_image=None,
    layout=None,
    license_image=None,
    max_results=None,
    page=1,
    output=None,
    download=False,
):
    """DuckDuckGo images search. Query params: https://duckduckgo.com/params

    Args:
        keywords (str): keywords for query.
        region (str, optional): wt-wt, us-en, uk-en, ru-ru, etc. Defaults to "wt-wt".
        safesearch (str, optional): on, moderate, off. Defaults to "moderate".
        time (Optional[str], optional): Day, Week, Month, Year. Defaults to None.
        size (Optional[str], optional): Small, Medium, Large, Wallpaper. Defaults to None.
        color (Optional[str], optional): color, Monochrome, Red, Orange, Yellow, Green, Blue,
            Purple, Pink, Brown, Black, Gray, Teal, White. Defaults to None.
        type_image (Optional[str], optional): photo, clipart, gif, transparent, line.
            Defaults to None.
        layout (Optional[str], optional): Square, Tall, Wide. Defaults to None.
        license_image (Optional[str], optional): any (All Creative Commons), Public (PublicDomain),
            Share (Free to Share and Use), ShareCommercially (Free to Share and Use Commercially),
            Modify (Free to Modify, Share, and Use), ModifyCommercially (Free to Modify, Share, and
            Use Commercially). Defaults to None.
        max_results (Optional[int], optional): maximum number of results, max=1000. Defaults to None.
            if max_results is set, then the parameter page is not taken into account.
        page (int, optional): page for pagination. Defaults to 1.
        output (Optional[str], optional): csv, json. Defaults to None.
        download (bool, optional): if True, download and save images to 'keywords' folder.
            Defaults to False.

    Returns:
        Optional[List[dict]]: DuckDuckGo text search results.
    """

    def get_ddg_images_page(page):
        payload["s"] = max(PAGINATION_STEP * (page - 1), 0)
        page_data = None
        try:
            resp = SESSION.get("https://duckduckgo.com/i.js", params=payload)
            resp.raise_for_status()
            page_data = resp.json().get("results", None)
        except Exception:
            logger.exception("")
            if not max_results:
                return None
        page_results = []
        if page_data:
            for row in page_data:
                if row["image"] not in cache:
                    cache.add(row["image"])
                    page_results.append(
                        {
                            "title": row["title"],
                            "image": row["image"],
                            "thumbnail": row["thumbnail"],
                            "url": row["url"],
                            "height": row["height"],
                            "width": row["width"],
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

    PAGINATION_STEP, MAX_API_RESULTS = 100, 1000

    # prepare payload
    safesearch_base = {"On": 1, "Moderate": 1, "Off": -1}
    time = f"time:{time}" if time else ""
    size = f"size:{size}" if size else ""
    color = f"color:{color}" if color else ""
    type_image = f"type:{type_image}" if type_image else ""
    layout = f"layout:{layout}" if layout else ""
    license_image = f"license:{license_image}" if license_image else ""
    payload = {
        "l": region,
        "o": "json",
        "s": max(PAGINATION_STEP * (page - 1), 0),
        "q": keywords,
        "vqd": vqd,
        "f": f"{time},{size},{color},{type_image},{layout},{license_image}",
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
                fs.append(executor.submit(get_ddg_images_page, page))
                sleep(min(iterations / 17, 0.4))  # sleep to prevent blocking
            for r in as_completed(fs):
                if r.result():
                    results.extend(r.result())
        results = results[:max_results]
    else:
        results = get_ddg_images_page(page=page)
        if not results:
            return None

    # save to csv or json file
    if output:
        _do_output("ddg_images", keywords, output, results)

    # download images
    if download:
        keywords = keywords.replace('"', "'")
        path = f"ddg_images_{keywords}_{datetime.now():%Y%m%d_%H%M%S}"
        os.makedirs(path, exist_ok=True)
        futures = []
        with ThreadPoolExecutor(30) as executor:
            for i, res in enumerate(results, start=1):
                filename = normalize("NFC", res["image"].split("/")[-1].split("?")[0])
                future = executor.submit(
                    _download_file, res["image"], path, f"{i}_{filename}"
                )
                futures.append(future)
            with progressbar(
                as_completed(futures),
                label="Downloading images",
                length=len(futures),
                show_percent=True,
                show_pos=True,
                width=0,
            ) as as_completed_futures:
                for i, future in enumerate(as_completed_futures, start=1):
                    logger.info("%s/%s", i, len(results))

    return results
