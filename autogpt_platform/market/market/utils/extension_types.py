import prisma.models


class AgentsWithRank(prisma.models.Agents):
    rank: float
