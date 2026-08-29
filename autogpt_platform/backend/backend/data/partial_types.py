import prisma.models


class StoreAgentWithRank(prisma.models.StoreAgent):
    rank: float
