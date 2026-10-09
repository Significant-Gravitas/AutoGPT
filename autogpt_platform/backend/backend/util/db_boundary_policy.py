"""The database implementation layer and the supported access gateways."""

# Existing query implementations whose historical filenames do not say "db".
# New query implementations belong in backend.data or a db.py / *_db.py module.
DATABASE_IMPLEMENTATIONS = frozenset(
    {
        "backend.api.features.experts.credentials",
        "backend.api.features.experts.credential_counts",
        "backend.api.features.experts.raise_attachments",
        "backend.api.features.experts.routines",
        "backend.api.features.experts.routine_jobs",
        "backend.api.features.experts.scheduling",
        "backend.api.features.experts.seed",
        "backend.api.features.experts.spend_approval",
        "backend.api.features.library._add_to_library",
        "backend.api.features.search.content_handlers",
        "backend.api.features.search.embeddings",
        "backend.api.features.search.hybrid_search",
        "backend.api.features.store.hybrid_search",
        "backend.api.features.store.skill_catalog",
        "backend.integrations.scoped_credentials",
    }
)

RAW_DATABASE_ACCESS = frozenset(
    {
        ".prisma",
        "prisma.Prisma",
        "prisma.Client",
        "prisma.client.Prisma",
        "prisma.client.Client",
        "prisma.client.get_client",
        "prisma.client.register",
        "prisma.get_client",
        "prisma.register",
        "backend.data.db.prisma",
        "backend.data.db.connect",
        "backend.data.db.disconnect",
        "backend.data.db.transaction",
        "backend.data.db.query_raw_with_schema",
        "backend.data.db.execute_raw_with_schema",
        "backend.data.credit.get_user_credit_model",
        "backend.data.credit.get_credit_model",
    }
)

CONNECTION_OWNERS = {
    "backend.cli.mailerlite_backfill": frozenset(
        {"mailerlite_backfill_command", "mailerlite_checkout_backfill_command"}
    ),
    "backend.cli.onboarding_role_backfill": frozenset(
        {"onboarding_role_backfill_command"}
    ),
}

CLI_DISPATCHER = "backend.cli.main"

DATABASE_GATEWAYS = (
    "backend.data.db_accessors.",
    "backend.util.clients.get_database_manager_async_client",
    "backend.util.clients.get_database_manager_client",
    "backend.data.db_manager.DatabaseManagerAsyncClient",
    "backend.data.db_manager.DatabaseManagerClient",
)


def is_database_implementation(module: str) -> bool:
    return (
        (module.startswith("backend.data.") and module != "backend.data.db_accessors")
        or module.rsplit(".", 1)[-1] == "db"
        or module.endswith("_db")
        or module in DATABASE_IMPLEMENTATIONS
    )


def is_gateway(target: str) -> bool:
    return any(
        (
            target.startswith(gateway)
            if gateway.endswith(".")
            else target == gateway or target.startswith(gateway + ".")
        )
        for gateway in DATABASE_GATEWAYS
    )


def is_connection_owner_dispatch(module: str, target: str) -> bool:
    return module == CLI_DISPATCHER and any(
        target == f"{owner}.{command}"
        for owner, commands in CONNECTION_OWNERS.items()
        for command in commands
    )
