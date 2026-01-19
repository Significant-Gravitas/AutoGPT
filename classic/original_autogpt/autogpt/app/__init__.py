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
        Path.cwd() / ".env",
        Path.home() / ".autogpt" / ".env",
        Path.home() / ".config" / "autogpt" / ".env",
        Path(__file__).parent.parent.parent / ".env",
    ]

    for env_path in env_locations:
        if env_path.exists():
            load_dotenv(env_path, verbose=True, override=True)
            break


_load_env_from_locations()

del _load_env_from_locations, load_dotenv, Path
