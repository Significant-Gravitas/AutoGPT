from prisma.models import Agents


class AgentsWithRank(Agents):
    rank: float
    search_text: str
