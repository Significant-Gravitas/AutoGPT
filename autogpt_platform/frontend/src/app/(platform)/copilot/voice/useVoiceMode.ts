"use client";

import type { UIMessage } from "ai";
import { useEffect, useRef, useState } from "react";

import { useToast } from "@/components/molecules/Toast/use-toast";

import {
  classifyUtterance,
  pickAcknowledgement,
  type UtteranceKind,
} from "./acknowledgements";
import {
  describeVoiceState,
  isMicOpen,
  voiceReduce,
  type VoiceEvent,
  type VoiceState,
} from "./micStateMachine";
import { takeVoiceStart } from "./pendingVoiceStart";
import { createReplyTextReader } from "./replyText";
import { synthesizeSpeech, transcribeUtterance } from "./speechApi";
import { LATER_CHUNK_MIN_CHARS, takeSpeakableChunks } from "./speechChunker";
import { createSpeechPlayer, type SpeechPlayer } from "./speechPlayer";
import { stripMarkdownForSpeech } from "./stripMarkdownForSpeech";
import { isRejectableTranscript } from "./transcriptFilters";
import { startVadSession, type VadSession } from "./vadSession";

/** The mic closes after this long without speech. */
export const SILENCE_TIMEOUT_MS = 8_000;
/** A session that never completes a turn — a noisy room — closes anyway. */
const MAX_SESSION_MS = 5 * 60 * 1000;
/** A reply that produces no speakable text still has to give the mic back. */
const REPLY_WATCHDOG_MS = 45_000;
/** A second, register-aware phrase for the long wait before the first token. */
const FOLLOW_UP_ACK_MS = 6_000;

interface Args {
  enabled: boolean;
  messages: UIMessage[];
  isStreaming: boolean;
  sessionId: string | null;
  silenceTimeoutMs?: number;
  onSend: (message: string) => void | Promise<void>;
}

export function useVoiceMode({
  enabled,
  messages,
  isStreaming,
  sessionId,
  silenceTimeoutMs = SILENCE_TIMEOUT_MS,
  onSend,
}: Args) {
  const [state, setState] = useState<VoiceState>("off");
  const { toast } = useToast();

  const inputs = useRef({ sessionId, onSend, silenceTimeoutMs });
  inputs.current = { sessionId, onSend, silenceTimeoutMs };

  const stateRef = useRef<VoiceState>("off");
  const playerRef = useRef<SpeechPlayer | null>(null);
  const vadRef = useRef<VadSession | null>(null);
  const timers = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());
  const reader = useRef(createReplyTextReader());
  const chunkBuffer = useRef("");
  const lastPhrase = useRef<string | null>(null);
  const replyDone = useRef(false);
  const spokeThisTurn = useRef(false);
  const wasStreaming = useRef(false);
  // Bumped by every activate and deactivate. Work started under an older
  // token belongs to a session the user has already left.
  const activation = useRef(0);
  // A ref as well as state: two clicks in one tick both read the pre-render
  // value, which is exactly the double-click that used to leak a session.
  const starting = useRef(false);
  const [isStarting, setIsStarting] = useState(false);

  useEffect(() => {
    if (stateRef.current === "thinking" || stateRef.current === "speaking") {
      consumeReply();
    }
  }, [messages]);

  // The stream ending is the cue to flush, never the cue to reopen the mic —
  // the reply finishes seconds before the speech does.
  useEffect(() => {
    const finished = wasStreaming.current && !isStreaming;
    wasStreaming.current = isStreaming;
    const speaking =
      stateRef.current === "thinking" || stateRef.current === "speaking";
    if (finished && speaking) finishReply();
  }, [isStreaming]);

  useEffect(() => {
    if (!enabled && stateRef.current !== "off") deactivate();
  }, [enabled]);

  // Asked for on the previous mount, from the composer of a chat that did not
  // exist yet. The audio element was already unlocked by that click.
  useEffect(() => {
    if (enabled && sessionId && takeVoiceStart()) void activate();
  }, [enabled, sessionId]);

  useEffect(() => teardown, []);

  return {
    state,
    isActive: state !== "off",
    isStarting,
    statusLabel: describeVoiceState(state),
    toggle,
    /** The visible stop button: cut the reply short and listen again. */
    interrupt,
  };

  function toggle() {
    if (stateRef.current === "off" && !starting.current) void activate();
    else deactivate();
  }

  async function activate() {
    const mine = ++activation.current;
    setStarting(true);
    // Unlocking here, inside the click, is what lets later chunks play at all.
    player().unlock();

    let session: VadSession;
    try {
      session = await startVadSession({
        onSpeechStart: () => dispatch({ type: "SPEECH_START" }),
        onMisfire: () => dispatch({ type: "SPEECH_MISFIRE" }),
        onSpeechEnd: (wav) => void handleUtterance(wav),
      });
    } catch (error) {
      report(error);
      return;
    } finally {
      setStarting(false);
    }

    // Model download and getUserMedia take seconds; the user may have left in
    // the meantime. An undestroyed session here keeps the mic live for good.
    if (mine !== activation.current) {
      void session.destroy();
      return;
    }

    vadRef.current = session;
    setTimer("session", MAX_SESSION_MS, deactivate);
    dispatch({ type: "ENABLE" });
  }

  function deactivate() {
    activation.current += 1;
    setStarting(false);
    clearTimers();
    playerRef.current?.stop();
    discardReply();
    void vadRef.current?.destroy();
    vadRef.current = null;
    dispatch({ type: "DISABLE" });
  }

  function interrupt() {
    playerRef.current?.stop();
    // Without this the held-back partial sentence is spoken at stream end,
    // with the mic already back open — the echo the state machine prevents.
    discardReply();
    dispatch({ type: "INTERRUPT" });
  }

  function setStarting(value: boolean) {
    starting.current = value;
    setIsStarting(value);
  }

  function discardReply() {
    chunkBuffer.current = "";
    reader.current.reset();
    replyDone.current = true;
  }

  async function handleUtterance(wav: Blob) {
    const mine = activation.current;
    dispatch({ type: "SPEECH_END" });
    // Spoken before the transcript exists, so the register is still unknown.
    acknowledge(null);

    let transcript = "";
    try {
      transcript = await transcribeUtterance(wav);
    } catch (error) {
      report(error);
      dispatch({ type: "TRANSCRIPT_DROPPED" });
      return;
    }

    // Transcription takes a second or two. Sending a turn the user opted out
    // of during it is worse than losing the utterance.
    if (mine !== activation.current || stateRef.current !== "transcribing") {
      playerRef.current?.stop();
      return;
    }

    if (isRejectableTranscript(transcript)) {
      playerRef.current?.stop();
      dispatch({ type: "TRANSCRIPT_DROPPED" });
      return;
    }

    startTurn(classifyUtterance(transcript));
    try {
      await inputs.current.onSend(transcript.trim());
    } catch (error) {
      report(error);
      dispatch({ type: "ERROR" });
    }
  }

  function consumeReply() {
    if (replyDone.current) return;
    const last = messages[messages.length - 1];
    if (last?.role !== "assistant") return;

    const full = last.parts
      .map((part) => (part.type === "text" ? part.text : ""))
      .join("");
    chunkBuffer.current += reader.current.read(full);

    // Only the first chunk races the clock; later ones read better long.
    const minChars = spokeThisTurn.current ? LATER_CHUNK_MIN_CHARS : 0;
    const { chunks, rest } = takeSpeakableChunks(
      chunkBuffer.current,
      false,
      minChars,
    );
    chunkBuffer.current = rest;
    chunks.forEach(speak);
  }

  function finishReply() {
    const tail = chunkBuffer.current + reader.current.flush();
    chunkBuffer.current = "";
    takeSpeakableChunks(tail, true).chunks.forEach(speak);

    replyDone.current = true;
    reader.current.reset();
    clearTimer("reply");
    if (player().isIdle()) dispatch({ type: "REPLY_DONE" });
  }

  function speak(text: string) {
    const speakable = stripMarkdownForSpeech(text);
    if (!speakable) return;
    spokeThisTurn.current = true;
    player().enqueue(speakable);
    dispatch({ type: "REPLY_SPEAKING" });
  }

  function startTurn(kind: UtteranceKind) {
    replyDone.current = false;
    spokeThisTurn.current = false;
    chunkBuffer.current = "";
    reader.current.reset();
    clearTimer("session");
    dispatch({ type: "TRANSCRIPT_SENT" });
    // AutoPilot's first token lands a median 13.9 s out; one more line keeps
    // the wait from reading as a dropped call.
    setTimer("followUpAck", FOLLOW_UP_ACK_MS, () => {
      if (!spokeThisTurn.current) acknowledge(kind);
    });
    setTimer("reply", REPLY_WATCHDOG_MS, () => {
      replyDone.current = true;
      if (player().isIdle()) dispatch({ type: "REPLY_DONE" });
    });
  }

  function acknowledge(kind: UtteranceKind | null) {
    const phrase = pickAcknowledgement(kind, lastPhrase.current);
    lastPhrase.current = phrase;
    player().enqueue(phrase, "acknowledgement");
  }

  function dispatch(event: VoiceEvent) {
    const next = voiceReduce(stateRef.current, event);
    if (next === stateRef.current) return;
    stateRef.current = next;
    setState(next);
    syncMic(next);
  }

  function syncMic(next: VoiceState) {
    if (isMicOpen(next)) vadRef.current?.resume();
    else vadRef.current?.pause();

    if (next === "listening") {
      setTimer("silence", inputs.current.silenceTimeoutMs, deactivate);
    } else {
      clearTimer("silence");
    }
    if (next !== "thinking") clearTimer("followUpAck");
    if (next === "listening" || next === "off") clearTimer("reply");
  }

  function player(): SpeechPlayer {
    if (!playerRef.current) {
      playerRef.current = createSpeechPlayer({
        synthesize: (text, kind) =>
          synthesizeSpeech(text, inputs.current.sessionId, kind),
        onIdle: () => {
          if (replyDone.current) dispatch({ type: "REPLY_DONE" });
        },
        onError: report,
      });
    }
    return playerRef.current;
  }

  function teardown() {
    clearTimers();
    playerRef.current?.destroy();
    void vadRef.current?.destroy();
  }

  function setTimer(name: string, ms: number, run: () => void) {
    clearTimer(name);
    timers.current.set(name, setTimeout(run, ms));
  }

  function clearTimer(name: string) {
    const timer = timers.current.get(name);
    if (timer) {
      clearTimeout(timer);
      timers.current.delete(name);
    }
  }

  function clearTimers() {
    timers.current.forEach(clearTimeout);
    timers.current.clear();
  }

  function report(error: unknown) {
    console.error("[Voice mode]", error);
    toast({
      title: "Voice mode",
      description:
        error instanceof Error ? error.message : "Something went wrong.",
      variant: "destructive",
    });
  }
}
