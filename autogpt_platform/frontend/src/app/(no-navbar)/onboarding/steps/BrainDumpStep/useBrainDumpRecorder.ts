// The recorder. Ordering is the contract: chunk → IndexedDB → upload
// queue. "Saved locally" only turns on once the IndexedDB write resolved,
// so the reassurance on screen is a fact rather than a hope.

import { useEffect, useRef, useState } from "react";
import { trackBrainDump } from "@/services/onboarding/brain-dump-analytics";
import {
  HARD_STOP_SECONDS,
  isPermissionDenied,
  newRecordingId,
  pickMimeType,
  TIMESLICE_MS,
} from "./helpers";
import { getMeta, getParts, savePart, saveMeta } from "./recordingStore";
import { buildPart, useUploadQueue } from "./useUploadQueue";

export type RecorderPhase = "idle" | "recording" | "stopping" | "stopped";

export function useBrainDumpRecorder() {
  const [phase, setPhase] = useState<RecorderPhase>("idle");
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [isSavedLocally, setIsSavedLocally] = useState(false);
  const [permissionDenied, setPermissionDenied] = useState(false);

  const recordingIdRef = useRef<string | null>(null);
  const mimeTypeRef = useRef<string>("audio/webm");
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const partIndexRef = useRef(0);
  const startedAtRef = useRef(0);
  // Mirrors `elapsedSeconds` because the state value a caller reads is the
  // one from its own render. `handleDone` awaits `stop()` — which itself
  // waits on the recorder and the pending IndexedDB writes — so by the
  // time it reports a duration, React has not re-rendered and the state
  // is short. The backend splits recordings over 20 minutes on this
  // number, so it has to be the real one.
  const elapsedSecondsRef = useRef(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // `ondataavailable` fires before `onstop`, and persisting is async — so
  // stopping has to wait on the in-flight writes or the final few seconds
  // of speech would be flushed to the server after we'd already decided
  // the queue was empty.
  const pendingWritesRef = useRef<Promise<void>[]>([]);

  const queue = useUploadQueue();

  useEffect(() => stopTracks, []);

  function stopTracks() {
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }

  async function start() {
    // Permission is requested on the first tap, never on screen load —
    // a browser prompt before the user has read the headline is how
    // denial rates go up.
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (error) {
      if (isPermissionDenied(error)) {
        setPermissionDenied(true);
        trackBrainDump("brain_dump_permission_denied");
      }
      return false;
    }

    const recordingId = newRecordingId();
    const mimeType = pickMimeType();
    recordingIdRef.current = recordingId;
    mimeTypeRef.current = mimeType;
    partIndexRef.current = 0;
    startedAtRef.current = Date.now();
    elapsedSecondsRef.current = 0;
    streamRef.current = stream;
    setElapsedSeconds(0);
    setIsSavedLocally(false);

    await rememberMeta({
      recordingId,
      mimeType,
      startedAt: startedAtRef.current,
      durationSecs: 0,
      finalized: false,
    });

    const recorder = new MediaRecorder(stream, { mimeType });
    recorder.ondataavailable = function handleChunk(event) {
      if (event.data.size === 0) return;
      pendingWritesRef.current.push(persistChunk(recordingId, event.data));
    };
    recorder.start(TIMESLICE_MS);
    recorderRef.current = recorder;

    timerRef.current = setInterval(tick, 250);
    setPhase("recording");
    trackBrainDump("brain_dump_started");
    return true;
  }

  async function persistChunk(recordingId: string, blob: Blob) {
    const part = buildPart(recordingId, partIndexRef.current, blob);
    partIndexRef.current += 1;
    try {
      await savePart(part);
      setIsSavedLocally(true);
    } catch {
      // No local persistence available (private-mode Safari, hardened
      // profiles). The upload queue becomes the only backup, so the chunk
      // still goes out — but "saved locally" stays off rather than lying.
    }
    queue.enqueue(part);
  }

  function tick() {
    const seconds = (Date.now() - startedAtRef.current) / 1000;
    elapsedSecondsRef.current = seconds;
    setElapsedSeconds(seconds);
    // 30 minutes stops the recorder but keeps every second captured —
    // the dump still submits, it just stops growing.
    if (seconds >= HARD_STOP_SECONDS) void stop();
  }

  async function stop(): Promise<number> {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === "inactive") {
      return elapsedSecondsRef.current;
    }
    setPhase("stopping");
    await new Promise<void>((resolve) => {
      recorder.onstop = () => resolve();
      recorder.stop();
    });
    await Promise.all(pendingWritesRef.current);
    pendingWritesRef.current = [];
    stopTracks();
    recorderRef.current = null;
    // Measured here rather than read off state: the awaits above mean the
    // last tick is already behind, and the tail of the take counts.
    const durationSecs = (Date.now() - startedAtRef.current) / 1000;
    elapsedSecondsRef.current = durationSecs;
    setElapsedSeconds(durationSecs);
    await rememberMeta({
      recordingId: recordingIdRef.current ?? "",
      mimeType: mimeTypeRef.current,
      startedAt: startedAtRef.current,
      durationSecs,
      finalized: false,
    });
    setPhase("stopped");
    return durationSecs;
  }

  // On mount, an unfinalized recording in IndexedDB means the last
  // session ended in a crash, a refresh or a back button. The parts are
  // still there, so the user is offered them back rather than told to
  // start over.
  async function findRecoverable() {
    const meta = await getMeta().catch(() => null);
    if (!meta || meta.finalized) return null;
    return meta;
  }

  // After a crash the upload queue is gone with the page, so the parts
  // have to be replayed out of IndexedDB. Parts already marked uploaded
  // are replayed too — the server keys them by index and overwrites, so a
  // duplicate is free while a missing one would punch a hole in the audio.
  async function adoptRecovered(recordingId: string, mimeType: string) {
    recordingIdRef.current = recordingId;
    mimeTypeRef.current = mimeType;
    const parts = await resendAllParts(recordingId);
    partIndexRef.current = parts.length;
    setPhase("stopped");
  }

  // Re-queue every part on disk, including ones already marked uploaded.
  //
  // The server's part buffer expires, so "we uploaded this once" is not
  // the same as "the server still has it" — and a missing part 0 means a
  // headless stream that can never be decoded. Rather than reason about
  // how long the buffer lives (clock skew, a constant duplicated on two
  // sides, a TTL someone later tunes), just re-send everything: the
  // server keys parts by index and overwrites, so a duplicate costs
  // bandwidth and nothing else. A whole dump is ~6 MB.
  async function resendAllParts(recordingId: string) {
    const parts = await getParts(recordingId).catch(() => []);
    parts.forEach((part) => queue.enqueue(part));
    return parts;
  }

  // Metadata is a convenience for crash recovery, not part of the audio —
  // failing to write it must never abort a take in progress.
  async function rememberMeta(meta: Parameters<typeof saveMeta>[0]) {
    await saveMeta(meta).catch(() => undefined);
  }

  return {
    phase,
    elapsedSeconds,
    isSavedLocally,
    permissionDenied,
    pendingUploads: queue.pendingCount,
    isOffline: queue.isOffline,
    audioStream: streamRef.current,
    recordingId: recordingIdRef.current,
    mimeType: mimeTypeRef.current,
    start,
    stop,
    flushUploads: queue.flush,
    resetQueue: queue.reset,
    findRecoverable,
    adoptRecovered,
    resendAllParts,
  };
}
