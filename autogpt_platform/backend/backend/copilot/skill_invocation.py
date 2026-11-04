"""Slash-command skill invocation: ``/skill-name arguments`` in a chat message.

Many catalog skills were written for Claude Code, where a user types
``/fix-issue 123`` and the skill's instructions arrive with ``123`` filled
in. This module parses that syntax and renders the skill body with the same
argument rules Claude Code documents at
https://code.claude.com/docs/en/skills.md, so those skills behave the same
here without editing their text:

- ``$ARGUMENTS`` is the whole argument string as typed.
- ``$ARGUMENTS[N]`` and ``$N`` are 0-based positional arguments, split with
  shell-style quoting (``"hello world"`` is one argument).
- Names from the ``arguments`` frontmatter map to positions in order, and a
  missing one expands to an empty string.
- A positional placeholder with no argument stays in the text unchanged.
- A single backslash escapes a placeholder (``\\$1.00`` renders ``$1.00``).
- Arguments that no placeholder received are appended as
  ``ARGUMENTS: <input>`` so the model still sees them.
- Argument values are inserted literally, never expanded again.
"""

import logging
import re
import shlex
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict

from backend.copilot.model import ChatMessage
from backend.copilot.service import (
    SKILL_INVOCATION_TAG,
    persist_current_user_message,
    split_leading_server_blocks,
    strip_server_injected_tags,
)
from backend.copilot.tools import skills as skill_registry
from backend.copilot.tools.skills import (
    LoadedSkillPackage,
    ParsedSkill,
    get_default_skill_with_body,
    is_skills_feature_enabled,
    list_user_skills,
    resolve_skill_scope,
)

logger = logging.getLogger(__name__)

_SLUG = re.compile(r"[a-z0-9_-]+")

# A command name: the skill slug, an upstream alias, or a plugin-namespaced
# name such as ``commercial-legal:amendment-history``.
_INVOCATION = re.compile(r"^/([A-Za-z0-9][A-Za-z0-9_.:-]*)(?:\s+(.*))?$", re.DOTALL)

# ``(\\*)`` captures the backslashes in front so a single one can escape the
# placeholder while ``\\`` and longer runs leave it live.
_PLACEHOLDER = re.compile(
    r"(\\*)\$(?:ARGUMENTS\[(\d+)\]|(ARGUMENTS)\b|(\d+)|([A-Za-z_][A-Za-z0-9_]*))"
)


class SkillInvocation(BaseModel):
    """A message that asks to run a skill by name."""

    model_config = ConfigDict(frozen=True)

    name: str
    arguments: str


def parse_skill_invocation(message: str) -> SkillInvocation | None:
    """The skill invocation a message starts with, or None.

    Only a message whose first character is ``/`` counts, so a path or URL
    later in ordinary prose never triggers a skill.
    """
    match = _INVOCATION.match(message.strip())
    if not match:
        return None
    return SkillInvocation(
        name=match.group(1), arguments=(match.group(2) or "").strip()
    )


def argument_names(frontmatter: Mapping[str, Any]) -> list[str]:
    """The ``arguments`` frontmatter as a list of names.

    Claude Code accepts a YAML list or a space-separated string.
    """
    value = frontmatter.get("arguments")
    if isinstance(value, str):
        return value.split()
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def split_arguments(arguments: str) -> list[str]:
    """Positional arguments, with shell-style quoting.

    Unbalanced quotes fall back to plain whitespace splitting rather than
    failing the whole invocation.
    """
    try:
        return shlex.split(arguments)
    except ValueError:
        return arguments.split()


def render_skill_body(body: str, arguments: str, names: list[str]) -> str:
    """The skill body with its argument placeholders filled in."""
    positional = split_arguments(arguments) if arguments else []
    named = {name: index for index, name in enumerate(names)}
    received = False

    def substitute(match: re.Match[str]) -> str:
        nonlocal received
        backslashes, bracket_index, whole, digit_index, name = match.groups()
        token = match.group(0)[len(backslashes) :]
        if name is not None and name not in named:
            # Not one of ours: leave ``$word`` and any backslash alone.
            return match.group(0)
        if len(backslashes) == 1:
            return token
        if whole is not None:
            received = True
            return backslashes + arguments
        index = bracket_index or digit_index
        if index is not None:
            position = int(index)
            if position >= len(positional):
                return match.group(0)
            received = True
            return backslashes + positional[position]
        # A declared name counts as receiving even when its position is empty.
        if name is None:
            return match.group(0)
        received = True
        position = named[name]
        value = positional[position] if position < len(positional) else ""
        return backslashes + value

    rendered = _PLACEHOLDER.sub(substitute, body)
    if arguments and not received:
        rendered = f"{rendered.rstrip()}\n\nARGUMENTS: {arguments}"
    return rendered


class InvocableSkill(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill: ParsedSkill
    # Built-in skills have no workspace folder to point the model at.
    is_default: bool
    # A user skill's package, copied into the turn's working directory the
    # way ``read_skill`` copies it; None for a built-in skill.
    package: LoadedSkillPackage | None = None


def get_persisted_skill_invocation(
    message: str | None, session_messages: list[ChatMessage]
) -> str | None:
    """Recover the current row's expansion when retrying the same command."""
    if not session_messages:
        return None
    current = session_messages[-1]
    if current.role != "user" or current.sequence is None or not current.content:
        return None
    prefix, text = split_leading_server_blocks(current.content)
    if f"<{SKILL_INVOCATION_TAG}>" not in prefix:
        return None
    if message and message != text:
        return None
    return current.content


async def resolve_invocable_skill(
    user_id: str, expert_id: str | None, name: str, session_id: str
) -> InvocableSkill | None:
    """The chat's skill a ``/name`` command runs, or None.

    Looks the name up as a slug among the session owner's skills, then the
    built-ins, then as an alias. A skill whose frontmatter sets
    ``user-invocable: false`` cannot be run this way. A user skill is loaded
    the way ``read_skill`` loads it, its package files copied into the turn's
    working directory included.
    """
    wanted = name.strip().lower()
    scope = await resolve_skill_scope(user_id, expert_id)
    owned = await list_user_skills(user_id, expert_id, scope)
    selected = next((entry for entry in owned if entry.name == wanted), None)
    found: InvocableSkill | None = None
    if selected is None and _SLUG.fullmatch(wanted):
        if skill := get_default_skill_with_body(wanted):
            found = InvocableSkill(skill=skill, is_default=True)
    if selected is None and found is None:
        selected = next((entry for entry in owned if wanted in entry.aliases), None)
    if selected is not None:
        found = InvocableSkill(skill=selected, is_default=False)
    if found is None or found.skill.extra.get("user-invocable") is False:
        return None
    if not found.is_default:
        manager, raw = await skill_registry.read_skill_for_load(
            user_id, expert_id, found.skill.name, scope
        )
        text = raw.decode("utf-8")
        parsed = skill_registry.parse_skill_markdown(
            text, fallback_name=found.skill.name
        )
        if parsed is None or parsed.extra.get("user-invocable") is False:
            return None
        try:
            loaded = await skill_registry.resolve_loaded_skill_version(
                user_id, expert_id, found.skill.name, text, manager=manager
            )
            package = await skill_registry.materialise_skill_package(
                manager,
                expert_id=expert_id,
                name=found.skill.name,
                session_id=session_id,
                version_files=loaded.files,
            )
        except (
            skill_registry.SkillUsePausedError,
            skill_registry.SkillWriteLockError,
            skill_registry.SkillPackageLoadError,
        ):
            return None
        await skill_registry.record_skill_loaded(
            user_id,
            expert_id,
            found.skill.name,
            version_id=loaded.version_id,
            session_id=session_id,
        )
        found = InvocableSkill(skill=parsed, is_default=False, package=package)
    return found


def render_invocation(found: InvocableSkill, invocation: SkillInvocation) -> str:
    """What the model reads for a ``/name`` command: the skill's
    instructions with the arguments filled in, and where its files are."""
    skill = found.skill
    body = render_skill_body(
        skill.body, invocation.arguments, argument_names(skill.extra)
    )
    lines = [
        f"The user ran /{invocation.name}, which loads the '{skill.name}' skill. "
        "Follow its instructions below for this request."
    ]
    package = found.package
    if package is not None:
        lines.extend(package.notes)
        if package.package_dir:
            # Claude Code's name for the skill's directory: here, the copy
            # in the shell's working directory that read_skill also makes.
            body = body.replace("${CLAUDE_SKILL_DIR}", package.package_dir)
            lines.append(
                f"If they are gone on a later turn, read_skill('{skill.name}') "
                "copies them there again."
            )
    return "\n".join(lines) + "\n\n" + strip_server_injected_tags(body).strip()


async def inject_skill_invocation(
    message: str,
    session_id: str,
    session_messages: list[ChatMessage],
    *,
    user_id: str | None,
    expert_id: str | None,
) -> str | None:
    """Expand a ``/name arguments`` command in the current user message.

    *message* is the current turn's user message as the model will see it,
    possibly already prefixed with server-injected context blocks. When its
    own text starts with a command naming one of the chat's skills, a
    ``<skill_invocation>`` block carrying the rendered skill goes in front of
    that text and the result is persisted as the turn's user row, so the
    skill stays in context and the chat still shows what the user typed.

    Returns the new message, or None when there is nothing to expand. A failed lookup
    leaves the message unchanged; persistence failures raise before sending.
    """
    if not user_id:
        return None
    prefix, text = split_leading_server_blocks(message)
    invocation = parse_skill_invocation(text)
    # A message rebuilt from history already carries its expansion.
    if invocation is None or f"<{SKILL_INVOCATION_TAG}>" in prefix:
        return None
    try:
        if not await is_skills_feature_enabled(user_id):
            return None
        found = await resolve_invocable_skill(
            user_id, expert_id, invocation.name, session_id
        )
    except Exception:
        logger.exception("[skills] could not resolve /%s", invocation.name)
        return None
    if found is None:
        return None
    block = (
        f"<{SKILL_INVOCATION_TAG}>\n"
        f"{render_invocation(found, invocation)}\n"
        f"</{SKILL_INVOCATION_TAG}>\n\n"
    )
    return await persist_current_user_message(
        session_id,
        session_messages,
        prefix + block + text,
        "inject_skill_invocation",
        require_persistence=True,
    )
