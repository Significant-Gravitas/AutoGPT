"""Bounded, reference-based host directory browsing for the control channel."""

from __future__ import annotations

import os
import re
import secrets
import time
import unicodedata
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import platform_info

MAX_DIRECTORY_ENTRIES = 200
MAX_SCANNED_ENTRIES = 1000
MAX_BROWSE_SESSIONS = 8
MAX_REFS_PER_BROWSE = 1200
MAX_CURSORS_PER_BROWSE = 32
BROWSE_TTL_SECONDS = 5 * 60

_WINDOWS_SPECIAL_NAMES = frozenset(
    {
        "$recycle.bin",
        "documents and settings",
        "program files",
        "program files (x86)",
        "programdata",
        "recovery",
        "system volume information",
        "windows",
    }
)


class DirectoryBrowseError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class DirectoryReference:
    directory_ref: str
    name: str
    path: str


@dataclass(frozen=True)
class DirectoryListing:
    browse_id: str
    current: DirectoryReference | None
    parent_ref: str | None
    entries: tuple[DirectoryReference, ...]
    next_cursor: str | None
    truncated: bool
    expires_at: float


@dataclass
class _ReferenceTarget:
    path: Path
    parent_ref: str | None


@dataclass(frozen=True)
class _PageCursor:
    directory_ref: str
    offset: int


@dataclass
class _BrowseSession:
    expires_at: float
    refs: dict[str, _ReferenceTarget] = field(default_factory=dict)
    cursors: OrderedDict[str, _PageCursor] = field(default_factory=OrderedDict)


class DirectoryBrowser:
    """Expose directories through short-lived opaque references.

    The platform never supplies a host path. Initial listing returns a virtual
    set of safe roots; every subsequent navigation and selection must use a
    reference minted on this process and control connection.
    """

    def __init__(
        self,
        *,
        platform_name: str | None = None,
        home: Path | None = None,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.platform_name = platform_name or platform_info.detect_platform()
        self.home = home or Path.home()
        self._now = now
        self._sessions: OrderedDict[str, _BrowseSession] = OrderedDict()

    def reset(self) -> None:
        """Invalidate every reference, including on a control-WS reconnect."""
        self._sessions.clear()

    def list_directories(
        self,
        browse_id: str | None,
        directory_ref: str | None,
        cursor: str | None = None,
    ) -> DirectoryListing:
        if (browse_id is None) != (directory_ref is None):
            raise DirectoryBrowseError(
                "DIRECTORY_REFERENCE_INVALID",
                "browse_id and directory_ref must both be null or both be set",
            )
        if cursor is not None and browse_id is None:
            raise DirectoryBrowseError(
                "DIRECTORY_REFERENCE_INVALID",
                "A page cursor requires a directory reference",
            )
        self._evict_expired()
        if browse_id is None:
            return self._start()
        assert directory_ref is not None
        session = self._get_session(browse_id)
        target = session.refs.get(directory_ref)
        if target is None:
            raise DirectoryBrowseError(
                "DIRECTORY_REFERENCE_INVALID",
                "The directory reference is invalid or expired",
            )
        offset = 0
        if cursor is not None:
            page = session.cursors.get(cursor)
            if page is None or page.directory_ref != directory_ref:
                raise DirectoryBrowseError(
                    "DIRECTORY_REFERENCE_INVALID",
                    "The directory page cursor is invalid or expired",
                )
            offset = page.offset
            session.cursors.move_to_end(cursor)
        current_path = self._canonical_accessible_directory(target.path)
        current = DirectoryReference(
            directory_ref=directory_ref,
            name=self._display_name(current_path),
            path=str(current_path),
        )
        entries, truncated, next_offset = self._list_children(
            session,
            current_path,
            directory_ref,
            offset,
        )
        next_cursor = (
            self._add_cursor(session, directory_ref, next_offset)
            if next_offset is not None
            else None
        )
        self._sessions.move_to_end(browse_id)
        return DirectoryListing(
            browse_id=browse_id,
            current=current,
            parent_ref=target.parent_ref,
            entries=tuple(entries),
            next_cursor=next_cursor,
            truncated=truncated,
            expires_at=session.expires_at,
        )

    def resolve(self, browse_id: str, directory_ref: str) -> Path:
        self._evict_expired()
        session = self._get_session(browse_id)
        target = session.refs.get(directory_ref)
        if target is None:
            raise DirectoryBrowseError(
                "DIRECTORY_REFERENCE_INVALID",
                "The directory reference is invalid or expired",
            )
        return self._canonical_accessible_directory(target.path)

    def resolve_and_consume(self, browse_id: str, directory_ref: str) -> Path:
        """Resolve one selected reference and invalidate its whole browse session."""
        resolved = self.resolve(browse_id, directory_ref)
        self.consume(browse_id)
        return resolved

    def consume(self, browse_id: str) -> None:
        self._sessions.pop(browse_id, None)

    def _start(self) -> DirectoryListing:
        while len(self._sessions) >= MAX_BROWSE_SESSIONS:
            self._sessions.popitem(last=False)
        browse_id = secrets.token_urlsafe(24)
        session = _BrowseSession(expires_at=self._now() + BROWSE_TTL_SECONDS)
        entries: list[DirectoryReference] = []
        for root in self._discover_roots():
            ref = self._add_ref(session, root, parent_ref=None)
            entries.append(
                DirectoryReference(
                    directory_ref=ref,
                    name=self._display_name(root),
                    path=str(root),
                )
            )
        self._sessions[browse_id] = session
        return DirectoryListing(
            browse_id=browse_id,
            current=None,
            parent_ref=None,
            entries=tuple(entries),
            next_cursor=None,
            truncated=False,
            expires_at=session.expires_at,
        )

    def _get_session(self, browse_id: str) -> _BrowseSession:
        session = self._sessions.get(browse_id)
        if session is None or session.expires_at <= self._now():
            self._sessions.pop(browse_id, None)
            raise DirectoryBrowseError(
                "DIRECTORY_REFERENCE_INVALID",
                "The directory reference is invalid or expired",
            )
        return session

    def _evict_expired(self) -> None:
        now = self._now()
        expired = [key for key, session in self._sessions.items() if session.expires_at <= now]
        for key in expired:
            self._sessions.pop(key, None)

    def _list_children(
        self,
        session: _BrowseSession,
        current: Path,
        current_ref: str,
        offset: int,
    ) -> tuple[list[DirectoryReference], bool, int | None]:
        candidates: list[Path] = []
        scanned = 0
        hard_truncated = False
        try:
            with os.scandir(current) as iterator:
                for entry in iterator:
                    scanned += 1
                    if scanned > MAX_SCANNED_ENTRIES:
                        hard_truncated = True
                        break
                    if not self._safe_directory_entry(entry):
                        continue
                    try:
                        child = self._canonical_accessible_directory(Path(entry.path))
                    except DirectoryBrowseError:
                        continue
                    candidates.append(child)
        except OSError as exc:
            raise DirectoryBrowseError(
                "DIRECTORY_UNAVAILABLE",
                "The selected directory is not accessible",
            ) from exc
        candidates.sort(key=lambda path: self._display_name(path).casefold())
        page = candidates[offset : offset + MAX_DIRECTORY_ENTRIES]
        next_offset = (
            offset + MAX_DIRECTORY_ENTRIES
            if len(candidates) > offset + MAX_DIRECTORY_ENTRIES
            else None
        )
        entries: list[DirectoryReference] = []
        for child in page:
            if len(session.refs) >= MAX_REFS_PER_BROWSE:
                hard_truncated = True
                next_offset = None
                break
            ref = self._add_ref(session, child, parent_ref=current_ref)
            entries.append(
                DirectoryReference(
                    directory_ref=ref,
                    name=self._display_name(child),
                    path=str(child),
                )
            )
        return entries, hard_truncated, next_offset

    @staticmethod
    def _add_cursor(
        session: _BrowseSession,
        directory_ref: str,
        offset: int,
    ) -> str:
        for token, page in session.cursors.items():
            if page.directory_ref == directory_ref and page.offset == offset:
                session.cursors.move_to_end(token)
                return token
        while len(session.cursors) >= MAX_CURSORS_PER_BROWSE:
            session.cursors.popitem(last=False)
        token = secrets.token_urlsafe(24)
        session.cursors[token] = _PageCursor(
            directory_ref=directory_ref,
            offset=offset,
        )
        return token

    def _safe_directory_entry(self, entry: os.DirEntry[str]) -> bool:
        try:
            if (
                entry.is_symlink()
                or _is_reparse_point(Path(entry.path))
                or not entry.is_dir(follow_symlinks=False)
            ):
                return False
        except OSError:
            return False
        if self.platform_name == "windows" and entry.name.casefold() in _WINDOWS_SPECIAL_NAMES:
            return False
        return _valid_wire_name(entry.name)

    def _canonical_accessible_directory(self, path: Path) -> Path:
        try:
            if path.is_symlink() or _is_reparse_point(path):
                raise DirectoryBrowseError(
                    "DIRECTORY_UNAVAILABLE",
                    "Symbolic links and reparse points cannot be browsed",
                )
            resolved = Path(os.path.realpath(path, strict=True))
            if not resolved.is_dir() or not os.access(resolved, os.R_OK | os.W_OK | os.X_OK):
                raise DirectoryBrowseError(
                    "DIRECTORY_UNAVAILABLE",
                    "The selected directory is not accessible",
                )
            return resolved
        except DirectoryBrowseError:
            raise
        except (OSError, RuntimeError) as exc:
            raise DirectoryBrowseError(
                "DIRECTORY_UNAVAILABLE",
                "The selected directory is not accessible",
            ) from exc

    def _discover_roots(self) -> tuple[Path, ...]:
        raw_roots = [self.home]
        if self.platform_name == "darwin":
            raw_roots.extend(self._children_of(Path("/Volumes")))
        elif self.platform_name == "windows":
            raw_roots.extend(_windows_drive_roots())
        elif self.platform_name == "wsl2":
            raw_roots.extend(self._wsl_drive_roots())
            raw_roots.extend(self._linux_user_mounts())
        elif self.platform_name == "linux":
            raw_roots.extend(self._linux_user_mounts())
        roots: list[Path] = []
        seen: set[str] = set()
        for raw in raw_roots:
            try:
                root = self._canonical_accessible_directory(raw)
            except DirectoryBrowseError:
                continue
            normalized = os.path.normcase(str(root))
            if normalized in seen or self._special_root(root):
                continue
            seen.add(normalized)
            roots.append(root)
            if len(roots) >= MAX_DIRECTORY_ENTRIES:
                break
        return tuple(roots)

    def _linux_user_mounts(self) -> list[Path]:
        username = self.home.name
        return [
            *self._children_of(Path("/media") / username),
            *self._children_of(Path("/run/media") / username),
        ]

    def _wsl_drive_roots(self) -> list[Path]:
        return [
            path for path in self._children_of(Path("/mnt")) if re.fullmatch(r"[A-Za-z]", path.name)
        ]

    @staticmethod
    def _children_of(parent: Path) -> list[Path]:
        children: list[Path] = []
        try:
            with os.scandir(parent) as iterator:
                for scanned, entry in enumerate(iterator, start=1):
                    if scanned > MAX_SCANNED_ENTRIES or len(children) >= MAX_DIRECTORY_ENTRIES:
                        break
                    try:
                        if (
                            entry.is_symlink()
                            or _is_reparse_point(Path(entry.path))
                            or not entry.is_dir(follow_symlinks=False)
                            or not _valid_wire_name(entry.name)
                        ):
                            continue
                    except OSError:
                        continue
                    children.append(Path(entry.path))
            return children
        except OSError:
            return []

    def _special_root(self, path: Path) -> bool:
        if self.platform_name in {"linux", "wsl2"}:
            return path in {Path("/dev"), Path("/proc"), Path("/sys"), Path("/run")}
        if self.platform_name == "darwin":
            return path in {Path("/System"), Path("/private")}
        return False

    @staticmethod
    def _add_ref(session: _BrowseSession, path: Path, parent_ref: str | None) -> str:
        ref = secrets.token_urlsafe(24)
        session.refs[ref] = _ReferenceTarget(path=path, parent_ref=parent_ref)
        return ref

    @staticmethod
    def _display_name(path: Path) -> str:
        return path.name or str(path)


def _valid_wire_name(name: str) -> bool:
    if not name or "\x00" in name:
        return False
    if any(unicodedata.category(char).startswith("C") for char in name):
        return False
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _is_reparse_point(path: Path) -> bool:
    """Detect Windows junctions/reparse points, including on Python 3.11."""
    native = getattr(os.path, "isjunction", None)
    if native is not None:
        try:
            if native(path):
                return True
        except OSError:
            return True
    if os.name != "nt":
        return False
    try:
        import ctypes

        attributes = int(ctypes.windll.kernel32.GetFileAttributesW(str(path)))  # type: ignore[attr-defined]
    except (AttributeError, OSError, TypeError, ValueError):
        return True
    invalid_attributes = 0xFFFFFFFF
    file_attribute_reparse_point = 0x0400
    return attributes == invalid_attributes or bool(attributes & file_attribute_reparse_point)


def _windows_drive_roots() -> list[Path]:
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        mask = int(kernel32.GetLogicalDrives())
        roots: list[Path] = []
        for index in range(26):
            if not mask & (1 << index):
                continue
            root = f"{chr(ord('A') + index)}:\\"
            drive_type = int(kernel32.GetDriveTypeW(root))
            if drive_type in {2, 3}:
                roots.append(Path(root))
        return roots
    except (AttributeError, OSError, TypeError, ValueError):
        return []


__all__ = [
    "BROWSE_TTL_SECONDS",
    "DirectoryBrowseError",
    "DirectoryBrowser",
    "DirectoryListing",
    "DirectoryReference",
    "MAX_BROWSE_SESSIONS",
    "MAX_DIRECTORY_ENTRIES",
    "MAX_REFS_PER_BROWSE",
    "MAX_SCANNED_ENTRIES",
]
