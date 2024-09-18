import typing

import fastapi
import prisma.enums

import market.db
import market.utils.extension_types

router = fastapi.APIRouter()


@router.get("/search")
async def search(
    query: str,
    page: int = fastapi.Query(1, description="The pagination page to start on"),
    page_size: int = fastapi.Query(
        10, description="The number of items to return per page"
    ),
    categories: typing.List[str] = fastapi.Query(
        None, description="The categories to filter by"
    ),
    description_threshold: int = fastapi.Query(
        60, description="The number of characters to return from the description"
    ),
    sort_by: str = fastapi.Query("rank", description="Sorting by column"),
    sort_order: typing.Literal["desc", "asc"] = fastapi.Query(
        "desc", description="The sort order based on sort_by"
    ),
    submission_status: prisma.enums.SubmissionStatus = fastapi.Query(
        None, description="The submission status to filter by"
    ),
) -> typing.List[market.utils.extension_types.AgentsWithRank]:
    """searches endpoint for agents

    Args:
        query (str): the search query
        page (int, optional): the pagination page to start on. Defaults to 1.
        page_size (int, optional): the number of items to return per page. Defaults to 10.
        category (str | None, optional): the agent category to filter by. None is no filter. Defaults to None.
        description_threshold (int, optional): the number of characters to return from the description. Defaults to 60.
        sort_by (str, optional): Sorting by column. Defaults to "rank".
        sort_order ('asc' | 'desc', optional): the sort order based on sort_by. Defaults to "desc".
    """
    return await market.db.search_db(
        query=query,
        page=page,
        page_size=page_size,
        categories=categories,
        description_threshold=description_threshold,
        sort_by=sort_by,
        sort_order=sort_order,
        submission_status=submission_status,
    )
