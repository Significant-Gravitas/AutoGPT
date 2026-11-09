"""A call's recording: the person's audio, kept until its transcript is made.

The relay writes every frame of the person's audio it takes to a spool file on
the pod's disk as it arrives. It uploads the spool to cloud storage every
``CHECKPOINT_S`` while the call runs and once more when it ends, so a pod that
dies loses at most the stretch since its last upload. Each upload is stored
the way a brain-dump recording is, with its marker, its expiry and its name
(``storage.store_audio``), so the brain dump's retention sweep deletes any
that outlives 48 hours (``retention.py``). The call's record names the latest
upload, so another pod can find it, and the uploads before it are deleted.
The post-call job deletes the last one once the transcript of record is
stored (:meth:`CallRecording.discard`). Nothing of Otto's is recorded.
"""

import asyncio
import io
import logging
import wave
from pathlib import Path
from typing import BinaryIO, Protocol

from backend.api.features.onboarding_dump import storage
from backend.api.features.onboarding_dump.models import RECORDING_FILENAME_PREFIX
from backend.util.cloud_storage import get_cloud_storage_handler
from backend.voice_relay.config import IN_RATE
from backend.voice_relay.locks import CallStore

logger = logging.getLogger(__name__)


class RecordingStore(Protocol):
    """Where recordings wait for their transcripts."""

    async def store(self, user_id: str, wav: bytes, filename: str) -> str:
        """Store *wav* for *user_id*. Returns its path."""
        ...

    async def fetch(self, path: str, user_id: str) -> bytes: ...

    async def delete(self, path: str) -> None:
        """Delete a recording. One already gone counts as deleted."""
        ...


class CloudRecordings:
    """The brain dump's storage for recordings: its private bucket, its
    recording marker and its expiry."""

    async def store(self, user_id: str, wav: bytes, filename: str) -> str:
        return await storage.store_audio(user_id, wav, filename)

    async def fetch(self, path: str, user_id: str) -> bytes:
        handler = await get_cloud_storage_handler()
        return await handler.retrieve_file(path, user_id=user_id)

    async def delete(self, path: str) -> None:
        await storage.delete_audio(path)


def recording_filename(call_id: str) -> str:
    """The name a call's recording is stored under: a brain-dump recording's,
    so the retention sweep finds it."""
    return f"{RECORDING_FILENAME_PREFIX}voice-{call_id}.wav"


def wav_of(pcm: bytes) -> bytes:
    """*pcm*, 16 kHz mono PCM16, as a WAV file."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(IN_RATE)
        out.writeframes(pcm)
    return buffer.getvalue()


def pcm_of(wav: bytes) -> bytes:
    """The audio of a call's recording. Raises ``ValueError`` for anything
    that is not 16 kHz mono PCM16 in a WAV file."""
    try:
        with wave.open(io.BytesIO(wav), "rb") as source:
            shape = (
                source.getnchannels(),
                source.getsampwidth(),
                source.getframerate(),
            )
            if shape != (1, 2, IN_RATE):
                raise ValueError(f"not a call recording: {shape}")
            return source.readframes(source.getnframes())
    except (wave.Error, EOFError) as error:
        raise ValueError("not a call recording") from error


async def delete_quietly(store: RecordingStore, path: str, call_id: str) -> bool:
    """Delete one recording. A failure is logged, never raised: the retention
    sweep deletes what is left. True when it went."""
    try:
        await store.delete(path)
    except Exception as error:
        logger.warning(
            f"Voice call {call_id}: a recording was not deleted, left to the "
            f"retention sweep ({type(error).__name__})"
        )
        return False
    return True


class CallRecording:
    """One call's recording, from its first frame to the deletion of its last
    upload.

    *call_store* is the call's record, which names the latest upload. An
    upload or a spool write that fails is logged and the call goes on: a call
    is never ended for its recording.
    """

    def __init__(
        self,
        *,
        call_id: str,
        user_id: str,
        spool_dir: str | Path,
        store: RecordingStore,
        call_store: CallStore,
    ) -> None:
        self.call_id = call_id
        self.user_id = user_id
        self.store = store
        self.call_store = call_store
        self.spool = Path(spool_dir) / f"{call_id}.pcm"
        # Samples of the person's audio recorded so far: where in the
        # recording the call is.
        self.samples = 0
        # Uploads not yet deleted, oldest first. The last one is the one the
        # call's record names, once naming it succeeded.
        self.uploads: list[str] = []
        self.closed = False
        self._file: BinaryIO | None = None
        self._lock = asyncio.Lock()
        self._checkpoint: asyncio.Task | None = None
        try:
            self.spool.parent.mkdir(parents=True, exist_ok=True)
            # Unbuffered, so every frame is on disk whole once written.
            self._file = open(self.spool, "ab", buffering=0)
        except OSError:
            logger.warning(
                f"Voice call {call_id}: no spool, nothing will be recorded",
                exc_info=True,
            )

    def append(self, pcm: bytes) -> None:
        """Record one frame of the person's audio."""
        if self._file is None:
            return
        try:
            self._file.write(pcm)
        except OSError:
            logger.warning(
                f"Voice call {self.call_id}: the spool stopped taking audio",
                exc_info=True,
            )
            self._close_file()
            return
        self.samples += len(pcm) // 2

    def checkpoint_soon(self) -> None:
        """Upload the recording so far in the background, unless an upload is
        already running."""
        if self.closed or (
            self._checkpoint is not None and not self._checkpoint.done()
        ):
            return
        self._checkpoint = asyncio.create_task(
            self.checkpoint(), name=f"voice-checkpoint-{self.call_id[:12]}"
        )

    async def checkpoint(self) -> bool:
        """Upload the recording so far, replacing the last upload. True once
        the call's record names it."""
        async with self._lock:
            if self.closed:
                return False
            try:
                return await self._upload(await self._read())
            except Exception:
                logger.warning(
                    f"Voice call {self.call_id}: checkpoint failed", exc_info=True
                )
                return False

    async def close(self) -> bytes:
        """Stop recording and upload all of it, then delete the spool if the
        upload went through. Returns the recording's audio."""
        async with self._lock:
            self.closed = True
            self._close_file()
            try:
                pcm = await self._read()
            except OSError:
                logger.warning(
                    f"Voice call {self.call_id}: the spool could not be read",
                    exc_info=True,
                )
                return b""
            if await self._upload(pcm):
                self._remove_spool()
            return pcm

    async def discard(self) -> None:
        """Delete every upload and the spool: the transcript is stored."""
        async with self._lock:
            self.closed = True
            self._close_file()
            uploads, self.uploads = self.uploads, []
            for path in uploads:
                await delete_quietly(self.store, path, self.call_id)
            self._remove_spool()

    async def discard_spool(self) -> None:
        """Delete the spool and keep the uploads, for another process to read
        the last one or for the retention sweep to delete."""
        async with self._lock:
            self.closed = True
            self._close_file()
            self._remove_spool()

    async def _read(self) -> bytes:
        size = self.samples * 2
        if size == 0:
            return b""
        return await asyncio.to_thread(self._read_spool, size)

    def _read_spool(self, size: int) -> bytes:
        with open(self.spool, "rb") as spool:
            return spool.read(size)

    async def _upload(self, pcm: bytes) -> bool:
        if not pcm:
            return False
        try:
            path = await self.store.store(
                self.user_id,
                await asyncio.to_thread(wav_of, pcm),
                recording_filename(self.call_id),
            )
        except Exception as error:
            logger.warning(
                f"Voice call {self.call_id}: recording not uploaded "
                f"({type(error).__name__})"
            )
            return False
        self.uploads.append(path)
        if not await self.call_store.record(audio_path=path):
            # The record still names an earlier upload, which stays until a
            # later upload is named.
            return False
        older, self.uploads = self.uploads[:-1], self.uploads[-1:]
        for stale in older:
            await delete_quietly(self.store, stale, self.call_id)
        return True

    def _close_file(self) -> None:
        file, self._file = self._file, None
        if file is not None:
            try:
                file.close()
            except OSError:
                logger.debug(f"Voice call {self.call_id}: spool did not close")

    def _remove_spool(self) -> None:
        try:
            self.spool.unlink(missing_ok=True)
        except OSError:
            logger.warning(f"Voice call {self.call_id}: spool not removed")
