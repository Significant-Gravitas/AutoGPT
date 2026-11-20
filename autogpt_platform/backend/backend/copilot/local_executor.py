"""Shared Local PC executor helpers."""

import unicodedata
from html import escape


class LocalPCExecutorMarker:
    pass


def build_local_pc_env_context(
    *, platform: str | None, arch: str | None, allowed_root: str | None
) -> str:
    """Build the trusted prompt block without trusting shim-provided text."""
    safe_platform = _escape_context_value(platform or "unknown")
    safe_arch = _escape_context_value(arch or "unknown")
    safe_root = _escape_context_value(allowed_root or "unknown")
    return "\n".join(
        (
            "execution_environment: local_pc",
            f"platform: {safe_platform}",
            f"architecture: {safe_arch}",
            f"working_dir: {safe_root}",
            "shell_scope: real user-level machine; not limited to working_dir",
        )
    )


def _escape_context_value(value: str) -> str:
    single_line = "".join(
        (
            " "
            if unicodedata.category(character).startswith("C")
            or character in {"\u2028", "\u2029"}
            else character
        )
        for character in value
    )
    return escape(single_line, quote=True)
