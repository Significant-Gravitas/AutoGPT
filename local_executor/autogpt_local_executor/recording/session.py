"""
RecordingSession — lifecycle, encrypted buffer, redaction, summary.

One session per recording. Responsibilities (docs/WORKFLOW_RECORDING.md):

- §6 lifecycle: start → consume steps → stop → summary; fetch returns the
  full recording post-redaction.
- §1: mint a shim-side `recording_id` ("rec_<uuid>"), never reused.
- §9 consent: a START without a valid shim-issued token is rejected by the
  caller (handler) using ConsentBroker; the session itself only runs once
  consent has been validated.
- §9 redaction: best-effort HYGIENE — secret-shaped fields get
  value.raw=null / value.type=secret. THIS IS NOT A PRIVACY GUARANTEE. OTPs,
  account numbers, SSNs-in-generic-fields will slip the deny-list. The real
  control is interpretation_route for pixel handling; the hygiene-redacted
  structured trajectory is fetched for authenticated browser review. Named as
  hygiene in code + docstrings on purpose.
- §9 at-rest: the buffer is encrypted on disk under a key DISTINCT from the
  audit-chain key (a different trust boundary), secure-erased on close unless
  pinned.
- §6 summary: enrichment_coverage counts (dom/ax/none) + duration.

Steps live in memory for the session and are mirrored to an encrypted on-disk
buffer so a long recording isn't held entirely in RAM and survives within the
process. The on-disk form is removed on close (secure-erase best-effort).
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
import uuid
from pathlib import Path

from ..protocol import (
    EnrichmentCoverage,
    InterpretationRoute,
    StepValue,
    TrajectoryStep,
    ValueType,
    WorkflowRecording,
)
from .crypto import RecordingCipher

logger = logging.getLogger(__name__)


# Best-effort secret-field hygiene deny-list (§9). Substring-matched against the
# step's enrichment label/role and the value.type. Deliberately conservative:
# this is hygiene, NOT a control. Anything not matched here (OTPs, account
# numbers, SSNs in generic fields) WILL pass through — interpretation_route is
# the real control.
_SECRET_LABEL_HINTS = (
    "password",
    "passwd",
    "pwd",
    "secret",
    "api key",
    "api_key",
    "apikey",
    "token",
    "credential",
    "private key",
    "cvv",
    "security code",
)
_SECRET_ROLES = ("securetextfield", "passwordfield", "password")


def mint_recording_id() -> str:
    """A fresh, never-reused recording id. UUID4 → astronomically collision-free."""
    return f"rec_{uuid.uuid4()}"


def _looks_like_secret(step: TrajectoryStep) -> bool:
    """Best-effort secret detection (HYGIENE, not a guarantee — §9).

    Flags a step's value when the declared value.type is `secret`, the
    a11y/DOM role is a password-style field, or the label matches a known
    secret hint. Generic-but-sensitive fields (SSN in a "Number" field, an OTP
    in a "Code" field) are NOT reliably caught — that gap is by design and
    documented; the privacy control is interpretation_route, not this list.
    """
    if step.value.type == "secret":
        return True
    role = (step.enrichment.role or "").lower().replace(" ", "")
    if role in _SECRET_ROLES:
        return True
    label = (step.enrichment.label or "").lower()
    return any(hint in label for hint in _SECRET_LABEL_HINTS)


def redact_step(step: TrajectoryStep) -> TrajectoryStep:
    """Apply best-effort secret hygiene to one step (§9).

    Returns the step unchanged when nothing looks secret. When it does, returns
    a copy with value.raw stripped to null, value.type set to `secret`, and
    redacted=True. Non-secret values — including non-secret PII — are LEFT
    INTACT. That is the known hygiene gap, not an oversight.
    """
    if not _looks_like_secret(step):
        return step
    new_type: ValueType = "secret"
    return step.model_copy(
        update={
            "value": StepValue(raw=None, type=new_type, is_parameter=step.value.is_parameter),
            "redacted": True,
        }
    )


class RecordingError(Exception):
    """Raised on misuse of a RecordingSession (e.g. fetch before stop)."""


class RecordingSession:
    """Holds one recording's lifecycle, buffer, and derived summary.

    Not thread-safe; driven from the daemon's single event loop. The handler
    enforces one-at-a-time (RECORDING_ALREADY_ACTIVE) above this class.
    """

    def __init__(
        self,
        *,
        machine_id: str,
        mode: str,
        interpretation_route: InterpretationRoute,
        channels: list[str],
        buffer_dir: Path,
        recording_id: str | None = None,
        cipher: RecordingCipher | None = None,
    ) -> None:
        self.recording_id = recording_id or mint_recording_id()
        self.machine_id = machine_id
        self.mode = mode
        self.interpretation_route = interpretation_route
        self.channels = list(channels)
        self._buffer_dir = Path(buffer_dir)
        self._cipher = cipher or RecordingCipher.create()
        self._steps: list[TrajectoryStep] = []
        self._created_at: float = time.time()
        self._started_at: float | None = None
        self._stopped_at: float | None = None
        # On-disk encrypted buffer path. Unique per session; never reused.
        self._buffer_path = self._buffer_dir / f"{self.recording_id}.{secrets.token_hex(8)}.enc"
        self._closed = False
        self._last_review_signature: tuple[tuple[int, ...], tuple[int, ...]] | None = None
        self._last_review_step_count: int | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Begin the session. Idempotent guard against double-start."""
        if self._started_at is not None:
            raise RecordingError(f"recording {self.recording_id} already started")
        self._started_at = time.time()
        self._buffer_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("RecordingSession %s started (mode=%s)", self.recording_id, self.mode)

    def append(self, step: TrajectoryStep) -> TrajectoryStep:
        """Buffer one captured step in memory + encrypted on disk.

        Returns the (redaction is NOT applied here — see `fetch`/`finalize`,
        which redact at read time so the in-memory/disk buffer keeps the raw
        capture under encryption until the user consents to send it).
        """
        if self._stopped_at is not None:
            raise RecordingError(f"recording {self.recording_id} is stopped")
        self._steps.append(step)
        self._persist_buffer()
        return step

    def stop(self) -> None:
        """Finalize the session. After stop, no more appends; fetch/summary OK."""
        if self._stopped_at is not None:
            return
        self._stopped_at = time.time()
        logger.debug("RecordingSession %s stopped (%d steps)", self.recording_id, len(self._steps))

    def close(self, *, pin: bool = False) -> None:
        """Secure-erase the on-disk buffer unless pinned (§9 at-rest).

        Best-effort overwrite-then-unlink. `pin=True` keeps the encrypted blob
        (the user chose to keep the recording).
        """
        self._closed = True
        if pin:
            return
        try:
            if self._buffer_path.is_file():
                size = self._buffer_path.stat().st_size
                with open(self._buffer_path, "r+b") as f:
                    f.write(secrets.token_bytes(size))
                    f.flush()
                    os.fsync(f.fileno())
                self._buffer_path.unlink()
        except OSError:
            logger.debug("secure-erase of recording buffer failed", exc_info=True)

    # ── Read-side (post-redaction) ────────────────────────────────────────

    @property
    def step_count(self) -> int:
        return len(self._steps)

    @property
    def is_active(self) -> bool:
        return self._started_at is not None and self._stopped_at is None

    def redacted_steps(self) -> list[TrajectoryStep]:
        """All buffered steps with best-effort hygiene applied (§9)."""
        return [redact_step(s) for s in self._steps]

    def to_recording(self) -> WorkflowRecording:
        """The full WorkflowRecording, post-redaction — what RECORDING_DATA carries."""
        return WorkflowRecording(
            recording_id=self.recording_id,
            created_at=self._created_at,
            machine_id=self.machine_id,
            interpretation_route=self.interpretation_route,
            steps=self.redacted_steps(),
            redaction_applied=True,
        )

    def apply_review(
        self,
        *,
        removed_step_seqs: list[int],
        redacted_step_seqs: list[int],
    ) -> int:
        """Apply the user's review decisions to the authoritative local copy."""
        if self._stopped_at is None:
            raise RecordingError("recording must be stopped before review")

        signature = (tuple(removed_step_seqs), tuple(redacted_step_seqs))
        if signature == self._last_review_signature:
            assert self._last_review_step_count is not None
            return self._last_review_step_count

        known = {step.seq for step in self._steps}
        requested = set(removed_step_seqs) | set(redacted_step_seqs)
        unknown = requested - known
        if unknown:
            raise RecordingError(f"review references unknown step sequence(s): {sorted(unknown)}")

        removed = set(removed_step_seqs)
        redacted = set(redacted_step_seqs) - removed
        reviewed: list[TrajectoryStep] = []
        for step in self._steps:
            if step.seq in removed:
                continue
            if step.seq in redacted:
                step = step.model_copy(
                    update={
                        "value": StepValue(
                            raw=None,
                            type="secret",
                            is_parameter=step.value.is_parameter,
                        ),
                        "redacted": True,
                    }
                )
            reviewed.append(step)
        self._steps = reviewed
        self._persist_buffer()
        self._last_review_signature = signature
        self._last_review_step_count = len(self._steps)
        return self._last_review_step_count

    def enrichment_coverage(self) -> EnrichmentCoverage:
        """Per-kind step counts for the summary (§6)."""
        dom = ax = none = 0
        for s in self._steps:
            kind = s.enrichment.kind
            if kind == "dom":
                dom += 1
            elif kind == "ax":
                ax += 1
            else:
                none += 1
        return EnrichmentCoverage(dom=dom, ax=ax, none=none)

    def duration_seconds(self) -> float:
        """Wall-clock from start to stop (or now if still running)."""
        if self._started_at is None:
            return 0.0
        end = self._stopped_at if self._stopped_at is not None else time.time()
        return max(end - self._started_at, 0.0)

    # ── Encrypted on-disk buffer ──────────────────────────────────────────

    def _persist_buffer(self) -> None:
        """Write the current step list to the encrypted on-disk buffer.

        Encrypted under the session cipher (distinct from the audit key, §9).
        Best-effort: a buffer write failure logs and continues — the in-memory
        list is the source of truth within the process.
        """
        try:
            self._buffer_dir.mkdir(parents=True, exist_ok=True)
            plaintext = json.dumps(
                [s.model_dump(mode="json") for s in self._steps],
                separators=(",", ":"),
            ).encode("utf-8")
            blob = self._cipher.encrypt(plaintext)
            tmp = self._buffer_path.with_suffix(self._buffer_path.suffix + ".tmp")
            with open(tmp, "wb") as f:
                f.write(blob)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._buffer_path)
            if os.name != "nt":
                try:
                    os.chmod(self._buffer_path, 0o600)
                except OSError:
                    pass
        except (OSError, ValueError):
            logger.debug("failed to persist recording buffer", exc_info=True)


__all__ = [
    "RecordingError",
    "RecordingSession",
    "mint_recording_id",
    "redact_step",
]
