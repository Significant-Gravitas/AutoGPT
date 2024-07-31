from prisma.models import Agents

Agents.create_partial(
    "AgentOnlyDescriptionNameAuthorIdCategories",
    include={"name", "author", "id", "categories"},
)
