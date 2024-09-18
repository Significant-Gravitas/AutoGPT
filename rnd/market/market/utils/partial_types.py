import prisma.models

prisma.models.Agents.create_partial(
    "AgentOnlyDescriptionNameAuthorIdCategories",
    include={"name", "author", "id", "categories"},
)
