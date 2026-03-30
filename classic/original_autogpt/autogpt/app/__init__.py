from pathlib import Path

from dotenv import load_dotenv


def _load_env_from_locations() -> None:
    """Load .env from multiple locations (first found wins).

    This allows running autogpt from any directory while still finding credentials.
    Search order:
        1. Current working directory
        2. User config (~/.autogpt/.env)
        3. XDG config (~/.config/autogpt/.env)
        4. Package installation directory
    """
    env_locations = [
        Path.cwd() / ".env",  # noqa: F821
        Path.home() / ".autogpt" / ".env",  # noqa: F821
        Path.home() / ".config" / "autogpt" / ".env",  # noqa: F821
        Path(__file__).parent.parent.parent / ".env",  # noqa: F821
    ]

    for env_path in env_locations:
        if env_path.exists():
            load_dotenv(env_path, verbose=True, override=True)  # noqa: F821
            break


_load_env_from_locations()

del _load_env_from_locations, load_dotenv, Path
