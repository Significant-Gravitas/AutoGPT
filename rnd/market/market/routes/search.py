from typing import List, Literal

from fastapi import APIRouter

from market.db import search_db
from market.utils.extension_types import AgentsWithRank

router = APIRouter()


@router.get("/search")
async def search(
    query: str,
    page: int = 1,
    page_size: int = 10,
    category: str | None = None,
    description_threshold: int = 60,
    sort_by: str = "createdAt",
    sort_order: Literal["desc"] | Literal["asc"] = "desc",
) -> List[AgentsWithRank]:
    """searches endpoint for agnets

    Args:
        query (str): the search query
        page (int, optional): the pagination page to start on. Defaults to 1.
        page_size (int, optional): the number of items to return per page. Defaults to 10.
        category (str | None, optional): the agent category to filter by. None is no filter. Defaults to None.
        description_threshold (int, optional): the number of characters to return from the description. Defaults to 60.
        sort_by (str, optional): Sorting options. Defaults to "createdAt".
        sort_order ('asc' | 'desc', optional): the sort order based on sort_by. Defaults to "desc".
    """
    return await search_db(
        query=query,
        page=page,
        page_size=page_size,
        category=category,
        description_threshold=description_threshold,
        sort_by=sort_by,
        sort_order=sort_order,
    )
