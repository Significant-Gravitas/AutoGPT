import click


@click.command(name="rotate-encryption-key")
@click.option(
    "--old-key",
    envvar="OLD_ENCRYPTION_KEY",
    required=True,
    help="The ENCRYPTION_KEY the data was written under. "
    "Also read from OLD_ENCRYPTION_KEY.",
)
@click.option("--apply", is_flag=True, help="Write the results. Without it, dry run.")
def rotate_encryption_key(old_key: str, apply: bool):
    """Re-encrypt stored credentials under the configured ENCRYPTION_KEY.

    Run it with the stack stopped, after changing ENCRYPTION_KEY: everything
    written under --old-key is decrypted and written back under the new key,
    so connected integrations keep working. Idempotent: a value already
    readable with the new key is never touched, and one that neither key can
    read is reported and left as it is.
    """
    import asyncio

    asyncio.run(_run_rotation(old_key=old_key, apply=apply))


async def _run_rotation(*, old_key: str, apply: bool) -> None:
    import prisma.models
    from cryptography.fernet import Fernet, InvalidToken, MultiFernet

    from backend.data.db import connect, disconnect
    from backend.util.settings import Settings

    new_key = Settings().secrets.encryption_key
    if not new_key:
        raise click.ClickException(
            "ENCRYPTION_KEY is not set, so there is no key to re-encrypt under."
        )
    if new_key == old_key:
        raise click.ClickException(
            "--old-key is the configured ENCRYPTION_KEY. Change ENCRYPTION_KEY "
            "first, then pass the previous value here."
        )
    try:
        new, old = Fernet(new_key.encode()), Fernet(old_key.encode())
    except ValueError as e:
        raise click.ClickException(f"Not a valid encryption key: {e}")
    rotator = MultiFernet([new, old])

    def reencrypt(token: str) -> str | None:
        """The token under the new key, None if it is already there."""
        try:
            new.decrypt(token.encode())
            return None
        except InvalidToken:
            return rotator.rotate(token.encode()).decode()

    async def user_tokens():
        rows = await prisma.models.User.prisma().find_many(
            where={"integrations": {"not": ""}}
        )
        return [(row.id, row.integrations) for row in rows]

    async def credential_tokens():
        rows = await prisma.models.IntegrationCredential.prisma().find_many()
        return [(row.id, row.encryptedPayload) for row in rows]

    async def bot_install_tokens():
        rows = await prisma.models.BotInstall.prisma().find_many()
        return [(row.id, row.credentials) for row in rows]

    async def write_user(id: str, token: str):
        await prisma.models.User.prisma().update(
            where={"id": id}, data={"integrations": token}
        )

    async def write_credential(id: str, token: str):
        await prisma.models.IntegrationCredential.prisma().update(
            where={"id": id}, data={"encryptedPayload": token}
        )

    async def write_bot_install(id: str, token: str):
        await prisma.models.BotInstall.prisma().update(
            where={"id": id}, data={"credentials": token}
        )

    targets = [
        ("User.integrations", user_tokens, write_user),
        ("IntegrationCredential.encryptedPayload", credential_tokens, write_credential),
        ("BotInstall.credentials", bot_install_tokens, write_bot_install),
    ]

    await connect()
    try:
        print(
            "Re-encrypting stored credentials"
            + ("" if apply else "  (dry run — nothing will be written)")
        )
        unreadable = 0
        for label, read, write in targets:
            rotated = current = failed = 0
            for id, stored in await read():
                if not stored:
                    continue
                try:
                    token = reencrypt(stored)
                except InvalidToken:
                    failed += 1
                    print(f"  {label} {id}: neither key can read it, left as is")
                    continue
                if token is None:
                    current += 1
                    continue
                rotated += 1
                if apply:
                    await write(id, token)
            unreadable += failed
            print(
                f"{label}: {rotated} {'re-encrypted' if apply else 'to re-encrypt'}, "
                f"{current} already on the new key, {failed} unreadable"
            )
        if unreadable:
            print(
                f"{unreadable} value(s) were written under some other key. "
                "Re-run with that key as --old-key, or reconnect those integrations."
            )
        if not apply:
            print("Re-run with --apply to write the changes.")
    finally:
        await disconnect()
