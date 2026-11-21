"""
Workflow-recording subsystem (shim side).

Records a workflow the user performs on their machine into a WorkflowRecording
(ordered TrajectorySteps) and hands it to the platform to generalize into a
reusable skill. See docs/WORKFLOW_RECORDING.md for the spec-locked contract.

Layout:
  - capture.py  — CaptureSource ABC (THE OS-hook seam) + Mock / floor / a11y.
  - session.py  — RecordingSession lifecycle, encrypted buffer, redaction, summary.
  - consent.py  — shim-issued, single-use consent tokens (§9).
  - route.py    — interpretation-route probing (§3.1).
  - crypto.py   — at-rest cipher under a key distinct from the audit key (§9).
"""

from .capture import (
    A11yEnricher,
    CaptureSource,
    MockCaptureSource,
    ScreenshotActionFloor,
)
from .consent import (
    ConsentBroker,
    ConsentError,
    request_user_consent,
)
from .route import RouteDecision, probe_interpretation_route
from .session import (
    RecordingError,
    RecordingSession,
    mint_recording_id,
    redact_step,
)

__all__ = [
    "A11yEnricher",
    "CaptureSource",
    "ConsentBroker",
    "ConsentError",
    "MockCaptureSource",
    "RecordingError",
    "RecordingSession",
    "RouteDecision",
    "ScreenshotActionFloor",
    "mint_recording_id",
    "probe_interpretation_route",
    "redact_step",
    "request_user_consent",
]
