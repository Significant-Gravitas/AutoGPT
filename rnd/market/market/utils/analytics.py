import prisma.models


async def track_download(agent_id: str):
    """
    Track the download event in the database.

    Args:
        agent_id (str): The ID of the agent.
        version (int | None, optional): The version of the agent. Defaults to None.

    Raises:
        Exception: If there is an error tracking the download event.
    """
    try:
        await prisma.models.AnalyticsTracker.prisma().upsert(
            where={"agentId": agent_id},
            data={
                "update": {"downloads": {"increment": 1}},
                "create": {"agentId": agent_id, "downloads": 1, "views": 0},
            },
        )
    except Exception as e:
        raise Exception(f"Error tracking download event: {str(e)}")


async def track_view(agent_id: str):
    """
    Track the view event in the database.

    Args:
        agent_id (str): The ID of the agent.
        version (int | None, optional): The version of the agent. Defaults to None.

    Raises:
        Exception: If there is an error tracking the view event.
    """
    try:
        await prisma.models.AnalyticsTracker.prisma().upsert(
            where={"agentId": agent_id},
            data={
                "update": {"views": {"increment": 1}},
                "create": {"agentId": agent_id, "downloads": 0, "views": 1},
            },
        )
    except Exception as e:
        raise Exception(f"Error tracking view event: {str(e)}")
