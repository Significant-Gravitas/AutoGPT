#!/usr/bin/env python3
"""Create and validate the persistent secrets used by the all-in-one image."""

from __future__ import annotations

import argparse
import base64
import binascii
import functools
import hashlib
import ipaddress
import os
import re
import secrets
import stat
import sys
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import NamedTuple
from urllib.parse import quote, unquote, urlsplit, urlunsplit

CONFIG_VERSION = "2"
SAFE_SECRET = re.compile(r"^[A-Za-z0-9._~-]+={0,2}$")
SAFE_USERNAME = re.compile(r"^[A-Za-z0-9._-]+$")
SAFE_DNS_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
# A ``NAME=`` line in a .env file, i.e. a secret left blank for setup to fill.
BLANK_ASSIGNMENT = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)=[ \t]*\r?\n?$")
# What Docker Compose's .env parser counts as blank: Go's unicode.IsSpace,
# line breaks included, and BLANKS, the ones within a line.
SPACES = "\t\n\v\f\r \x85\xa0" + "".join(
    chr(code)
    for code in (0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F, 0x205F, 0x3000)
)
BLANKS = "\t\v\f\r \x85\xa0"
EXPORT = re.compile(r"export[\t\n\f\r ]+")
# A line break or other control character, which no service password carries.
CONTROL = re.compile(r"[\x00-\x1f\x7f]")
# The escapes Docker Compose decodes in a double-quoted .env value, besides
# octal `\0ddd`. It keeps the backslash of any other. `\$` becomes `$$`,
# which interpolation then reads as `$`.
ESCAPES = {
    "a": "\a",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
    '"': '"',
    "\\": "\\",
    "$": "$$",
}
ESCAPE = re.compile(r"\\(?:[abfnrtv\"\\$]|0[0-9]{0,3})")
# What Docker Compose substitutes in an unquoted or double-quoted .env value,
# as compose-go's template pattern finds it: `$$` by `$`, and `$NAME` or
# `${NAME...}` by the environment the stack starts in. A name is what Go's
# case-insensitive `[_a-z]` takes, the Kelvin sign and the long s included. In
# `${...}` it stands alone or is followed by `-`, `+` or `?` (each optionally
# after `:`) and a default, alternative or message that runs to the last `}`
# on the line; a `${` that opens no name is a template Compose refuses.
NAME = r"[_A-Za-z\u017f\u212a][_A-Za-z0-9\u017f\u212a]*"
TEMPLATE = re.compile(
    r"\$(?:(?P<escaped>\$)|(?P<named>"
    + NAME
    + r")|\{(?:(?P<braced>"
    + NAME
    + r"(?::?[-+?](?P<nested>.*))?)\}|(?P<invalid>)))"
)
# Characters that a password can't carry into a connection string as is, or
# that Docker Compose would read as a variable reference.
UNSAFE_IN_URL = re.compile(r"[@/?#%$\[\]\s\"'\\`\x00-\x1f\x7f]")
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
LOCAL_DATABASE_HOSTS = LOCAL_HOSTS | {"db"}
# Each service password's host settings in backend/.env, in the order the
# backend reads them (it takes REDIS_CLUSTER_HOST over REDIS_HOST), and the
# host names that mean the service the stack bundles: this machine, or its
# Compose name. The password of a service that runs anywhere else is that
# service's, so setup neither generates nor replaces it.
BUNDLED_SERVICE_HOSTS = {
    "DB_PASS": (("DB_HOST",), LOCAL_DATABASE_HOSTS),
    "REDIS_PASSWORD": (
        ("REDIS_CLUSTER_HOST", "REDIS_HOST"),
        LOCAL_HOSTS | {"redis-0", "redis-1", "redis-2"},
    ),
    "RABBITMQ_DEFAULT_PASS": (
        ("RABBITMQ_CLUSTER_HOST", "RABBITMQ_HOST"),
        LOCAL_HOSTS | {"rabbitmq"},
    ),
    "GRAPHITI_FALKORDB_PASSWORD": (
        ("GRAPHITI_FALKORDB_HOST",),
        LOCAL_HOSTS | {"falkordb"},
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    ensure_parser = subparsers.add_parser("ensure")
    ensure_parser.add_argument("--path", type=Path, required=True)

    fill_parser = subparsers.add_parser("fill-env")
    fill_parser.add_argument("--path", type=Path, action="append", required=True)
    fill_parser.add_argument("--missing-ok", action="store_true")

    check_parser = subparsers.add_parser("check-env-defaults")
    check_parser.add_argument("--root", type=Path, required=True)

    url_parser = subparsers.add_parser("validate-public-url")
    url_parser.add_argument("url")

    args = parser.parse_args()
    try:
        if args.command == "ensure":
            ensure_runtime_config(args.path, os.environ)
        elif args.command == "fill-env":
            for path in args.path:
                if args.missing_ok and not path.exists():
                    continue
                if path.is_symlink():
                    print(f"{path}: symlink, left untouched", file=sys.stderr)
                    continue
                names = fill_env_blanks(path)
                if names:
                    print(f"{path}: generated {', '.join(names)}")
                if _shell_overrides_database_password(path, os.environ):
                    print(
                        f"{path}: POSTGRES_PASSWORD is also set in this shell, and "
                        "Docker Compose takes it over this file, so the database "
                        "and the services would get that value instead. Unset it "
                        "before `docker compose up`.",
                        file=sys.stderr,
                    )
        elif args.command == "check-env-defaults":
            offenders = check_env_defaults(args.root)
            for location in offenders:
                print(
                    f"{location}: ships a working secret value. .env.default is "
                    "public, so this value is published. Leave it blank and let "
                    "`make init-env` generate one.",
                    file=sys.stderr,
                )
            if offenders:
                return 1
        else:
            print(validate_public_url(args.url))
    except (OSError, ValueError) as exc:
        print(f"runtime configuration error: {exc}", file=sys.stderr)
        return 2
    return 0


def ensure_runtime_config(path: Path, environment: Mapping[str, str]) -> dict[str, str]:
    """Return the existing secrets or atomically create them on first boot.

    A version 1 config, written before INTERNAL_API_KEY existed, gets one. It
    is saved as the current version once it passes every check, so a boot that
    fails them leaves the file as it was."""
    if path.is_symlink():
        raise ValueError(f"refusing symlink at {path}")
    if path.exists():
        values = _read_config(path)
        upgrade = values.get("AUTOGPT_RUNTIME_CONFIG_VERSION") == "1"
        if upgrade:
            values = _upgraded_from_version_1(values, environment)
        _validate_values(values)
        _verify_environment_matches(values, environment)
        if upgrade:
            _write_config(path, values)
        path.chmod(0o600)
        return values

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    values = _new_values(environment)
    _write_config(path, values)
    return values


def fill_env_blanks(path: Path) -> list[str]:
    """Give every blank secret in a .env file a freshly generated value.

    Idempotent by construction: re-running setup never rotates a value an
    operator (or an earlier run) already put there. Passwords of the stack's
    own services (`CREDENTIALS`) go further, because each service reads its
    password from these files whenever it starts, so a new one costs only a
    restart: one the file's .env.default declares is added if the file predates
    it, and one that still holds a published value is replaced. That holds for
    the services the stack bundles only: a password for a service backend/.env
    points at another host is that service's, and stays as it is. The database
    password lives in three files and is filled from one value. The file is
    read the way Docker Compose reads it, and only values change in it.
    Returns the names that were given a value.
    """
    if path.is_symlink():
        raise ValueError(f"refusing symlink at {path}")
    text = _read(path)
    settings = _parse_settings(path, text)
    _check_passwords_are_plain(path, settings)
    database_password = functools.lru_cache(maxsize=None)(
        lambda: _shared_database_password(path)
    )

    filled: list[str] = []
    replacements: list[tuple[Setting, str]] = []
    for setting in settings:
        value = _new_value(setting, text, path, database_password)
        if value is not None:
            replacements.append((setting, value))
            filled.append(setting.name)
    # From the end, so each value's offsets still hold when its turn comes.
    for setting, value in reversed(replacements):
        text = text[: setting.start] + value + text[setting.end :]

    added: list[tuple[str, str]] = []
    for name in _missing_credentials(path, {setting.name for setting in settings}):
        value = _new_value(Setting(name, "", True, 0, 0), "", path, database_password)
        if value is not None:
            added.append((name, value))
    if added:
        if text and not text.endswith("\n"):
            text += "\n"
        text += "# Generated for this install by `make init-env`.\n"
        text += "".join(f"{name}={value}\n" for name, value in added)
        filled.extend(name for name, _ in added)

    if path.parent.name not in ("backend", "frontend"):
        # Compose puts this one into every connection string as it is.
        password = _effective(_parse_settings(path, text)).get("POSTGRES_PASSWORD")
        _check_database_password(password.value if password else "")
    if filled:
        _replace_atomically(path, text)
    return filled


def _shell_overrides_database_password(
    path: Path, environment: Mapping[str, str]
) -> bool:
    """Whether Docker Compose, started from `environment`, would put that
    environment's POSTGRES_PASSWORD in place of the one in `path`, the stack's
    own .env: Compose takes a variable the shell sets over that file."""
    if path.parent.name in ("backend", "frontend"):
        return False
    if "POSTGRES_PASSWORD" not in environment:
        return False
    setting = _file_settings(path).get("POSTGRES_PASSWORD")
    return setting is None or setting.value != environment["POSTGRES_PASSWORD"]


def _check_passwords_are_plain(path: Path, settings: list[Setting]) -> None:
    """Refuse a service password Docker Compose works out only when the stack
    starts, from an environment this can't see (`$NAME`, `${NAME}`, or the
    name alone on a line), so it could come out blank or short, and one with a
    line break or other control character, which no service can take."""
    effective = _effective(settings)
    for name in sorted(CREDENTIALS - {"DATABASE_URL"}):
        setting = effective.get(name)
        if setting is None or (setting.literal and not CONTROL.search(setting.value)):
            continue
        if name in BUNDLED_SERVICE_HOSTS and not _uses_bundled_service(name, effective):
            continue  # The password of a service of the backend's own.
        raise ValueError(
            f"{name} in {path} is not a plain value: Docker Compose takes it "
            "from the environment when the stack starts (a $ reference, or the "
            "name alone on a line), so it could come out blank or short, or it "
            "has a line break or other control character. Write the password "
            "itself (in single quotes if it has a $ in it), or clear it and run "
            "this again to generate one."
        )


def _new_value(
    setting: Setting, text: str, path: Path, database_password: Callable[[], str]
) -> str | None:
    """What `fill_env_blanks` writes in place of a setting's value as written,
    or None to keep it."""
    name, value = setting.name, setting.value
    if not setting.literal:
        return None
    if name in ("DATABASE_URL", "DIRECT_URL"):
        written = text[setting.start : setting.end]
        return _database_url(written, value, path, database_password)
    if name in BUNDLED_SERVICE_HOSTS and not _uses_bundled_service(
        name, _file_settings(path)
    ):
        return None  # The password of a service of the backend's own.
    if name in DATABASE_PASSWORD_SETTINGS:
        return database_password() if _is_unset(value) else None
    generator = SECRET_GENERATORS.get(name)
    if generator is None:
        return None
    if not value or (name in CREDENTIALS and _is_published(value)):
        return generator()
    return None


def _database_url(
    written: str, value: str, path: Path, database_password: Callable[[], str]
) -> str | None:
    """frontend/.env's blank DATABASE_URL, or a connection string to the local
    database that still carries a published password, pointed at the shared
    database password. Only the password changes; quotes and the rest of the
    value stay as they were written."""
    if not value:
        if path.parent.name != "frontend":
            return None
        return FRONTEND_DATABASE_URL.format(
            password=quote(database_password(), safe="")
        )
    password, host = _password_and_host(value)
    if (
        password is None
        or not _is_local_host(host)
        or not _is_published(unquote(password))
    ):
        return None
    published = f":{password}@"
    if published not in written:
        return None
    new = quote(database_password(), safe="")
    return written.lstrip(BLANKS).replace(published, f":{new}@", 1)


def _shared_database_password(path: Path) -> str:
    password = (
        _existing_database_password(path) or SECRET_GENERATORS["POSTGRES_PASSWORD"]()
    )
    _check_database_password(password)
    return password


def _existing_database_password(path: Path) -> str | None:
    """The database password another of the platform's .env files already
    holds, so a blank or published copy takes it instead of a new value."""
    parent = path.parent
    root = parent.parent if parent.name in ("backend", "frontend") else parent
    for value in _database_password_copies(root):
        if not _is_unset(value):
            return value
    return None


def _database_password_copies(root: Path) -> Iterator[str]:
    """The bundled database's password as each .env file holds it. backend/.env
    and frontend/.env count only while they point at the bundled database: a
    database of their own has a password that is not ours to copy. Neither
    does a value Compose works out only when the stack starts."""
    backend = _file_settings(root / "backend" / ".env")
    copies = [_file_settings(root / ".env").get("POSTGRES_PASSWORD")]
    if _uses_bundled_service("DB_PASS", backend):
        copies.append(backend.get("DB_PASS"))
    for setting in copies:
        if setting is not None and setting.literal:
            yield setting.value
    url = _file_settings(root / "frontend" / ".env").get("DATABASE_URL")
    if url is not None and url.literal:
        password, host = _password_and_host(url.value)
        if password is not None and _is_local_host(host):
            yield unquote(password)


def _check_database_password(password: str) -> None:
    if UNSAFE_IN_URL.search(password):
        raise ValueError(
            "the database password in autogpt_platform/.env, backend/.env or "
            "frontend/.env has a character a connection string can't carry as "
            "is (@ / ? # % $, a quote, a backslash, a space or a control "
            "character). Change it in all three to letters, digits, -, _, . "
            "and ~, or clear all three and run this again to generate one."
        )


def _uses_bundled_service(name: str, settings: Mapping[str, Setting]) -> bool:
    """Whether backend/.env, as Compose reads it, points the service that
    password `name` logs in to at the one the stack bundles. The first of its
    host settings with a value decides, as in the backend; with none, it is
    the bundled one, the backend's default. A host Compose works out only when
    the stack starts does not count as that."""
    host_settings, bundled_hosts = BUNDLED_SERVICE_HOSTS[name]
    for host_setting in host_settings:
        host = settings.get(host_setting)
        if host is None or (host.literal and not host.value):
            continue
        return host.literal and host.value.lower().rstrip(".") in bundled_hosts
    return True


def _password_and_host(url: str) -> tuple[str | None, str | None]:
    """A connection string's password, still percent-encoded, and its host;
    Nones for one urllib can't take apart."""
    try:
        parts = urlsplit(url)
        return parts.password, parts.hostname
    except ValueError:
        return None, None


def _is_local_host(host: str | None) -> bool:
    return not host or host.lower().rstrip(".") in LOCAL_DATABASE_HOSTS


class Setting(NamedTuple):
    """A `NAME=value` statement of a .env file, as Docker Compose reads it."""

    name: str
    # Its value, with any `$` reference left as written.
    value: str
    # False if Compose works the value out only when the stack starts: it
    # fills in a `$NAME` or `${NAME}`, or the name stands alone on a line.
    literal: bool
    # Where the value as written begins, just after the `=`, and ends: after
    # the closing quote, or before an unquoted value's ` # comment`.
    start: int
    end: int


def _file_settings(path: Path) -> dict[str, Setting]:
    """What Compose takes from a .env file: the last setting of each name. A
    symlinked file is read, although `fill_env_blanks` never writes through
    one."""
    if not path.is_file():
        return {}
    return _effective(_parse_settings(path, _read(path)))


def _read(path: Path) -> str:
    """A .env file byte for byte, line endings included, as Compose reads it."""
    return path.read_bytes().decode("utf-8")


def _parse_settings(path: Path, text: str) -> list[Setting]:
    try:
        return _settings(text)
    except ValueError as exc:
        raise ValueError(f"Docker Compose can't read {path}: {exc}") from None


def _effective(settings: list[Setting]) -> dict[str, Setting]:
    """Of several settings of one name, Compose takes the last."""
    return {setting.name: setting for setting in settings}


def _settings(text: str) -> list[Setting]:
    """The statements of a .env file, read as Docker Compose reads the whole
    file: past a byte-order mark, blank lines and `#` lines; a name, indented
    or after `export `, then `=` or `:`; a quoted value that may run on over
    lines, and may have another statement after it on the same line; a name
    alone on a line, which takes the environment's value. Raises ValueError
    where Compose refuses the file."""
    settings: list[Setting] = []
    position = 1 if text.startswith("\ufeff") else 0
    while (position := _statement_start(text, position)) < len(text):
        if EXPORT.match(text, position):
            position += len("export")
            while position < len(text) and text[position] in BLANKS:
                position += 1
        name_start = position
        while position < len(text) and text[position] not in "=:\n":
            character = text[position]
            if not (character in BLANKS or character in "_.-[]" or character.isalnum()):
                raise ValueError(f"unexpected character {character!r} in a name")
            position += 1
        if position == len(text):
            break  # Compose files the last word of a file under no name.
        name = text[name_start:position].rstrip(SPACES)
        if " " in name:
            raise ValueError(f"a name with a space in it: {name!r}")
        separator = text[position]
        position += 1
        if separator == "\n":
            settings.append(Setting(name, "", False, position - 1, position - 1))
            continue
        start = position
        while position < len(text) and text[position] in BLANKS:
            if text.startswith("\r\n", position):
                break  # Blank to Compose too, but it ends the line: leave it be.
            position += 1
        value, literal, end, position = _value(text, position)
        settings.append(Setting(name, value, literal, start, end))
    return settings


def _statement_start(text: str, position: int) -> int:
    """Where the next statement begins: past blanks, line breaks and `#` lines."""
    while True:
        while position < len(text) and text[position] in SPACES:
            position += 1
        if position >= len(text) or text[position] != "#":
            return position
        line_end = text.find("\n", position)
        position = len(text) if line_end == -1 else line_end


def _value(text: str, position: int) -> tuple[str, bool, int, int]:
    """The value at `position`: what Compose takes, whether it is literal (see
    `Setting`), where it ends as written, and where the next statement may
    begin. Single-quoted values are literal, and a backslash escapes only the
    quote. Double-quoted values decode `\\n`, `\\t`, `\\"`, `\\\\`, `\\$` and the
    like, then are interpolated, as are unquoted ones, which end at ` #`."""
    quote_character = text[position : position + 1]
    if quote_character not in ("'", '"'):
        line_end = text.find("\n", position)
        line_end = len(text) if line_end == -1 else line_end
        written = text[position:line_end].split(" #", 1)[0].rstrip(SPACES)
        value, literal = _interpolate(written)
        return value, literal, position + len(written), min(line_end + 1, len(text))
    characters: list[str] = []
    escaped = False
    for index in range(position + 1, len(text)):
        character = text[index]
        if character == quote_character and not escaped:
            value = "".join(characters)
            if quote_character == "'":
                return value, True, index + 1, index + 1
            value, literal = _interpolate(ESCAPE.sub(_unescape, value))
            return value, literal, index + 1, index + 1
        if character == quote_character:
            characters.append(character)
        elif escaped:
            characters.append("\\" + character)
        elif character == "\\":
            escaped = True
            continue
        else:
            characters.append(character)
        escaped = False
    raise ValueError("a quoted value with no closing quote")


def _unescape(match: re.Match[str]) -> str:
    """One escape, as Compose decodes it with Go's strconv.UnquoteChar after
    turning `\\0ddd` into `\\ddd`: an octal one needs all three digits."""
    escape = match.group(0)
    if escape[1] != "0":
        return ESCAPES[escape[1]]
    digits = escape[2:]
    if len(digits) == 3 and set(digits) <= set("01234567") and int(digits, 8) < 256:
        return chr(int(digits, 8))
    return "\\" + digits


def _interpolate(value: str) -> tuple[str, bool]:
    """`value` with each `$$` read as `$`, and False if anything else in it
    is one of Compose's variable references. Raises ValueError for a `${`
    Compose can't read, which makes it refuse the whole file."""
    _check_templates(value)
    literal = True

    def substitute(match: re.Match[str]) -> str:
        nonlocal literal
        if match.group("escaped"):
            return "$"
        literal = False
        return match.group(0)

    return TEMPLATE.sub(substitute, value), literal


def _check_templates(text: str) -> None:
    """Raise ValueError for a template in `text` that compose-go's
    template.Substitute refuses, read step for step as it reads them: a
    `${...}` is cut at the `}` that closes it, or runs to the last `}` on its
    line if none does, and its default, alternative or message is read the
    same way, like whatever follows the cut. Compose reads a default only
    when the environment calls for it; this reads every one, so it can refuse
    a template Compose would read, never the other way round."""
    for match in TEMPLATE.finditer(text):
        if match.group("escaped") or match.group("named"):
            continue
        template = match.group(0)
        close = _closing_brace(template)
        if close >= 0:
            _check_templates(template[close + 1 :])
            template = template[: close + 1]
        braced = TEMPLATE.match(template)
        if braced is None or braced.group("braced") is None:
            raise ValueError("a ${ that starts no variable it can read")
        if braced.group("nested"):
            _check_templates(braced.group("nested"))


def _closing_brace(template: str) -> int:
    """Where compose-go's getFirstBraceClosingIndex puts the end of the
    template: the `}` that leaves no brace open, where every `{` opens one
    and the character after it is skipped; -1 if no `}` does."""
    depth = 0
    index = 0
    while index < len(template):
        if template[index] == "}":
            depth -= 1
            if depth == 0:
                return index
        if template[index] == "{":
            depth += 1
            index += 1
        index += 1
    return -1


def _missing_credentials(path: Path, assigned: set[str]) -> list[str]:
    """Credentials the file's .env.default leaves for setup to fill that the
    file has no line for, typically because it was copied before they existed.
    Without them the services would fall back to a blank .env.default value."""
    defaults = path.with_name(".env.default")
    if path.name == defaults.name or not defaults.is_file():
        return []
    names = [
        match.group("name")
        for line in defaults.read_text(encoding="utf-8").splitlines(keepends=True)
        if (match := BLANK_ASSIGNMENT.match(line)) is not None
    ]
    return [name for name in names if name in CREDENTIALS and name not in assigned]


def _is_unset(value: str) -> bool:
    return not value or _is_published(value)


def _is_published(value: str) -> bool:
    return hashlib.sha256(value.encode()).hexdigest() in PUBLISHED_CREDENTIAL_DIGESTS


def check_env_defaults(root: Path) -> list[str]:
    """Return `path:NAME` for every retired secret not assigned exactly once, blank."""
    offenders: list[str] = []
    for relative_path, names in BLANK_IN_ENV_DEFAULT.items():
        content = (root / relative_path).read_text(encoding="utf-8")
        for name in names:
            values = re.findall(rf"(?m)^{re.escape(name)}=([^\r\n]*)\r?$", content)
            if values != [""]:
                offenders.append(f"{relative_path}:{name}")
    return offenders


def _replace_atomically(path: Path, content: str) -> None:
    """Rewrite `path` owner-only, with no window where it is truncated or
    half-written. The file now holds generated secrets, so it stops inheriting
    whatever mode the copied `.env.default` had."""
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def validate_public_url(value: str) -> str:
    """Validate and normalize the externally reachable origin."""
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("AUTOGPT_PUBLIC_URL must use http or https")
    if not parsed.hostname:
        raise ValueError("AUTOGPT_PUBLIC_URL must include a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("AUTOGPT_PUBLIC_URL must not include credentials")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError(
            "AUTOGPT_PUBLIC_URL must be an origin without path, query, or fragment"
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("AUTOGPT_PUBLIC_URL has an invalid port") from exc

    host = _normalize_public_host(parsed.hostname)
    netloc = host if port is None else f"{host}:{port}"
    return urlunsplit((parsed.scheme, netloc, "", "", ""))


def _normalize_public_host(host: str) -> str:
    """Return a config-safe IP literal or IDNA DNS name."""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        try:
            ascii_host = host.rstrip(".").encode("idna").decode("ascii").lower()
        except UnicodeError as exc:
            raise ValueError("AUTOGPT_PUBLIC_URL has an invalid host") from exc
        labels = ascii_host.split(".")
        if (
            not ascii_host
            or len(ascii_host) > 253
            or any(not SAFE_DNS_LABEL.fullmatch(label) for label in labels)
        ):
            raise ValueError("AUTOGPT_PUBLIC_URL has an invalid host")
        return ascii_host
    return f"[{address.compressed}]" if address.version == 6 else address.compressed


def _new_values(environment: Mapping[str, str]) -> dict[str, str]:
    vapid_private, vapid_public = _configured_or_generated_vapid(environment)
    values = {
        "AUTOGPT_RUNTIME_CONFIG_VERSION": CONFIG_VERSION,
        "RABBITMQ_DEFAULT_USER": environment.get("RABBITMQ_DEFAULT_USER") or "autogpt",
        **{
            name: _configured_or_generated(environment, name, generator)
            for name, generator in SECRET_GENERATORS.items()
        },
        "VAPID_PRIVATE_KEY": vapid_private,
        "VAPID_PUBLIC_KEY": vapid_public,
    }
    _validate_values(values)
    return values


def _configured_or_generated(
    environment: Mapping[str, str], name: str, generator: Callable[[], str]
) -> str:
    return environment.get(name) or generator()


def _upgraded_from_version_1(
    values: dict[str, str], environment: Mapping[str, str]
) -> dict[str, str]:
    """A version 1 config is a version 2 config without INTERNAL_API_KEY.

    This step produces version 2, whatever the current version is: a later
    version's own step takes it from there."""
    if "INTERNAL_API_KEY" in values:
        raise ValueError("runtime configuration has missing or unknown keys")
    return {
        **values,
        "AUTOGPT_RUNTIME_CONFIG_VERSION": "2",
        "INTERNAL_API_KEY": _configured_or_generated(
            environment, "INTERNAL_API_KEY", SECRET_GENERATORS["INTERNAL_API_KEY"]
        ),
    }


def _configured_or_generated_vapid(
    environment: Mapping[str, str],
) -> tuple[str, str]:
    private = environment.get("VAPID_PRIVATE_KEY")
    public = environment.get("VAPID_PUBLIC_KEY")
    if bool(private) != bool(public):
        raise ValueError("VAPID_PRIVATE_KEY and VAPID_PUBLIC_KEY must be set together")
    return (private, public) if private and public else _generate_vapid_keypair()


def _generate_vapid_keypair() -> tuple[str, str]:
    try:
        from cryptography.hazmat.primitives.asymmetric import ec
    except ImportError as exc:
        raise ValueError("cryptography is required to generate VAPID keys") from exc

    key = ec.generate_private_key(ec.SECP256R1())
    numbers = key.private_numbers()
    private = numbers.private_value.to_bytes(32, "big")
    public_numbers = numbers.public_numbers
    public = (
        b"\x04"
        + public_numbers.x.to_bytes(32, "big")
        + public_numbers.y.to_bytes(32, "big")
    )
    return _base64url(private), _base64url(public)


def _fernet_key() -> str:
    return base64.urlsafe_b64encode(os.urandom(32)).decode("ascii")


# The single source of truth for "this setting is a secret, and this is how a
# local one is made". `.env.default` ships these blank (a working value in a
# public file is a published value); every bootstrap path fills them from here.
SECRET_GENERATORS: dict[str, Callable[[], str]] = {
    "POSTGRES_PASSWORD": lambda: secrets.token_urlsafe(36),
    "RABBITMQ_DEFAULT_PASS": lambda: secrets.token_urlsafe(36),
    "REDIS_PASSWORD": lambda: secrets.token_urlsafe(36),
    "BETTER_AUTH_SECRET": lambda: secrets.token_urlsafe(48),
    "ENCRYPTION_KEY": _fernet_key,
    "UNSUBSCRIBE_SECRET_KEY": lambda: secrets.token_urlsafe(36),
    "GRAPHITI_FALKORDB_PASSWORD": lambda: secrets.token_urlsafe(36),
    "INTERNAL_API_KEY": lambda: secrets.token_urlsafe(32),
}

# Passwords of the stack's own services. Each service takes its password from
# the .env files whenever it starts (the database's is applied on every boot by
# db/entrypoint.sh), so `fill-env` may add one or replace a published one: a
# restart is all it costs. Keys that encrypt or sign stored data are never
# replaced, since data written under the old key would become unreadable.
CREDENTIALS = frozenset(
    {
        "POSTGRES_PASSWORD",
        "DB_PASS",
        "DATABASE_URL",
        "RABBITMQ_DEFAULT_PASS",
        "REDIS_PASSWORD",
        "GRAPHITI_FALKORDB_PASSWORD",
    }
)

# The database password: POSTGRES_PASSWORD in autogpt_platform/.env, which
# Compose gives the `db` service and every service's connection string, and its
# copies for running outside Docker, DB_PASS in backend/.env and the password
# in frontend/.env's DATABASE_URL. They are filled from one value.
DATABASE_PASSWORD_SETTINGS = ("POSTGRES_PASSWORD", "DB_PASS")
FRONTEND_DATABASE_URL = "postgresql://postgres:{password}@localhost:5432/postgres"

# SHA-256 of every value the public .env.default and .env.example files have
# given a credential. They are published, so `fill-env` treats them as blank.
# Re-derive one from git history rather than writing the value down, e.g.
#   git log -p -- autogpt_platform/backend/.env.default | grep '^+DB_PASS='
PUBLISHED_CREDENTIAL_DIGESTS = frozenset(
    {
        "53d3eb3d9500cdcdbbe39c030eb2959438bcc459d5de9380db2a3c85b5de5360",  # pragma: allowlist secret
        "9b8769a4a742959a2d0298c36fb70623f2dfacda8436237df08d8dfd5b37374c",  # pragma: allowlist secret
        "e4b879cfddda5d56e0d4c5b1ec46e605159469234adbabc5c5bcf2ebce5755cd",  # pragma: allowlist secret
        "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",  # pragma: allowlist secret
        "d417f13ced355a54a9444aed43c7ba4f47a81c0c07b8fedf3207cc876bcb8db3",  # pragma: allowlist secret
    }
)

# Secrets already retired from the public .env.default files (SECRT-2611).
# `check-env-defaults` keeps them blank: entropy scanners do not reliably catch
# a freshly generated key pasted back into one of these lines, so the invariant
# is asserted directly. Paths are relative to autogpt_platform/.
BLANK_IN_ENV_DEFAULT: dict[str, tuple[str, ...]] = {
    ".env.default": ("POSTGRES_PASSWORD",),
    "backend/.env.default": (
        "DB_PASS",
        "RABBITMQ_DEFAULT_PASS",
        "REDIS_PASSWORD",
        "ENCRYPTION_KEY",
        "UNSUBSCRIBE_SECRET_KEY",
        "INTERNAL_API_KEY",
        "GRAPHITI_FALKORDB_PASSWORD",
    ),
    "frontend/.env.default": ("BETTER_AUTH_SECRET", "DATABASE_URL"),
}


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _validate_values(values: dict[str, str]) -> None:
    expected = {
        "AUTOGPT_RUNTIME_CONFIG_VERSION",
        "POSTGRES_PASSWORD",
        "RABBITMQ_DEFAULT_USER",
        "RABBITMQ_DEFAULT_PASS",
        "REDIS_PASSWORD",
        "BETTER_AUTH_SECRET",
        "ENCRYPTION_KEY",
        "UNSUBSCRIBE_SECRET_KEY",
        "GRAPHITI_FALKORDB_PASSWORD",
        "INTERNAL_API_KEY",
        "VAPID_PRIVATE_KEY",
        "VAPID_PUBLIC_KEY",
    }
    if set(values) != expected:
        raise ValueError("runtime configuration has missing or unknown keys")
    if values["AUTOGPT_RUNTIME_CONFIG_VERSION"] != CONFIG_VERSION:
        raise ValueError("unsupported runtime configuration version")
    if not SAFE_USERNAME.fullmatch(values["RABBITMQ_DEFAULT_USER"]):
        raise ValueError("RABBITMQ_DEFAULT_USER contains unsupported characters")
    if values["RABBITMQ_DEFAULT_USER"] == "guest":
        raise ValueError("RABBITMQ_DEFAULT_USER may not use the built-in guest account")

    for name in expected - {"AUTOGPT_RUNTIME_CONFIG_VERSION", "RABBITMQ_DEFAULT_USER"}:
        value = values[name]
        if len(value) < 32 or not SAFE_SECRET.fullmatch(value):
            raise ValueError(f"{name} must be at least 32 URL-safe characters")

    try:
        decoded_key = base64.urlsafe_b64decode(values["ENCRYPTION_KEY"])
    except (ValueError, binascii.Error) as exc:
        raise ValueError("ENCRYPTION_KEY must be a Fernet-compatible key") from exc
    if len(decoded_key) != 32:
        raise ValueError("ENCRYPTION_KEY must decode to exactly 32 bytes")

    vapid_private = _decode_base64url(values["VAPID_PRIVATE_KEY"])
    vapid_public = _decode_base64url(values["VAPID_PUBLIC_KEY"])
    if len(vapid_private) != 32:
        raise ValueError("VAPID_PRIVATE_KEY must decode to exactly 32 bytes")
    if len(vapid_public) != 65 or vapid_public[0] != 4:
        raise ValueError("VAPID_PUBLIC_KEY must be an uncompressed P-256 public key")


def _decode_base64url(value: str) -> bytes:
    padded = value + "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(padded)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid URL-safe base64 value") from exc


def _read_config(path: Path) -> dict[str, str]:
    mode = path.stat().st_mode
    if not stat.S_ISREG(mode):
        raise ValueError(f"runtime configuration is not a regular file: {path}")

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="ascii").splitlines(), 1
    ):
        if not raw_line or raw_line.startswith("#"):
            continue
        name, separator, value = raw_line.partition("=")
        if not separator or not name or name in values:
            raise ValueError(f"invalid runtime configuration line {line_number}")
        values[name] = value
    return values


def _verify_environment_matches(
    values: dict[str, str], environment: Mapping[str, str]
) -> None:
    for name, persisted in values.items():
        configured = environment.get(name)
        if configured and configured != persisted:
            raise ValueError(
                f"{name} differs from the value persisted on first boot; "
                "restore the original value or start with a new data volume"
            )


def _write_config(path: Path, values: dict[str, str]) -> None:
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        try:
            stream = os.fdopen(descriptor, "w", encoding="ascii")
        except BaseException:
            os.close(descriptor)
            raise
        with stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write("# Generated once by the AutoGPT all-in-one image.\n")
            for name, value in values.items():
                stream.write(f"{name}={value}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        if hasattr(os, "O_CLOEXEC"):
            directory_flags |= os.O_CLOEXEC
        directory_descriptor = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
