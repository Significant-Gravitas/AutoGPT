"use client";

import type { UIMessage } from "ai";
import { useEffect, useRef, useState } from "react";

import { useToast } from "@/components/molecules/Toast/use-toast";
import { trackVoiceMode } from "@/services/copilot/voice-mode-analytics";

import { playClickSound, primeClickSound } from "./clickSound";
import {
  describeVoiceState,
  isMicOpen,
  voiceReduce,
  type VoiceEvent,
  type VoiceState,
} from "./micStateMachine";
import { setVoiceTurnActive, takeVoiceStart } from "./pendingVoiceStart";
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
/**
 * Nothing at all from the turn for this long — no text, no tool activity —
 * and the mic goes back. Reset on every sign of life: a tool chain routinely
 * runs longer than this, and giving up mid-turn strands the reply unspoken.
 */
const REPLY_SILENCE_MS = 90_000;
/**
 * A sentence at the very end of the buffer is held back in case the next
 * delta turns "3." into "3.5". If no more text arrives for this long the
 * turn has moved on to tool calls, and holding it means silence for as long
 * as the tools take — so say it.
 */
const TEXT_SETTLED_MS = 800;

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
  const replyDone = useRef(false);
  const spokeThisTurn = useRef(false);
  const lastMessageId = useRef("");
  const turnIndex = useRef(0);
  /** Speech end, for the two latencies the funnel measures. */
  const utteranceEndedAt = useRef(0);
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
    /** The visible stop button: cut the reply short and leave voice mode. */
    stop: () => deactivate("user"),
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
    primeClickSound();

    let session: VadSession;
    try {
      session = await startVadSession({
        onSpeechStart: () => dispatch({ type: "SPEECH_START" }),
        onMisfire: () => {
          trackVoiceMode("voice_turn_dropped", { reason: "vad_misfire" });
          dispatch({ type: "SPEECH_MISFIRE" });
        },
        onSpeechEnd: (wav) => void handleUtterance(wav),
      });
    } catch (error) {
      report(error);
      const denied =
        error instanceof DOMException && error.name === "NotAllowedError";
      trackVoiceMode(
        denied ? "voice_mode_permission_denied" : "voice_mode_error",
        { stage: "vad_start" },
      );
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
    setVoiceTurnActive(true);
    turnIndex.current = 0;
    trackVoiceMode("voice_mode_started", {
      entry: inputs.current.sessionId ? "existing_chat" : "new_chat",
    });
    setTimer("session", MAX_SESSION_MS, () => deactivate("silence_timeout"));
    dispatch({ type: "ENABLE" });
  }

  function deactivate(reason: "user" | "silence_timeout" = "user") {
    if (stateRef.current !== "off") {
      trackVoiceMode(
        reason === "silence_timeout"
          ? "voice_mode_timed_out"
          : "voice_mode_stopped",
        { turns: turnIndex.current, state: stateRef.current },
      );
    }
    activation.current += 1;
    setVoiceTurnActive(false);
    setStarting(false);
    clearTimers();
    playerRef.current?.stop();
    discardReply();
    void vadRef.current?.destroy();
    vadRef.current = null;
    dispatch({ type: "DISABLE" });
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
    utteranceEndedAt.current = Date.now();
    dispatch({ type: "SPEECH_END" });
    playClickSound();

    let transcript = "";
    try {
      transcript = await transcribeUtterance(wav);
    } catch (error) {
      report(error);
      trackVoiceMode("voice_turn_dropped", { reason: "transcribe_failed" });
      dispatch({ type: "TRANSCRIPT_DROPPED" });
      return;
    }
    trackVoiceMode("voice_transcribe_latency_ms", {
      ms: Date.now() - utteranceEndedAt.current,
    });

    // Transcription takes a second or two. Sending a turn the user opted out
    // of during it is worse than losing the utterance.
    if (mine !== activation.current || stateRef.current !== "transcribing") {
      playerRef.current?.stop();
      return;
    }

    if (isRejectableTranscript(transcript)) {
      playerRef.current?.stop();
      trackVoiceMode("voice_turn_dropped", { reason: "filler_or_empty" });
      dispatch({ type: "TRANSCRIPT_DROPPED" });
      return;
    }

    startTurn();
    turnIndex.current += 1;
    trackVoiceMode("voice_turn_sent", {
      turn_index: turnIndex.current,
      transcript_chars: transcript.trim().length,
    });
    try {
      await inputs.current.onSend(transcript.trim());
    } catch (error) {
      report(error);
      trackVoiceMode("voice_mode_error", { stage: "send" });
      dispatch({ type: "ERROR" });
    }
  }

  function consumeReply() {
    if (replyDone.current) return;
    // Any activity at all — a tool part, a status, more text — proves the
    // turn is alive, so the mic is not owed back yet.
    armReplyWatchdog();

    const last = messages[messages.length - 1];
    if (last?.role !== "assistant") return;

    // A message after a tool round is a fresh passage: its first sentence
    // races the clock again rather than waiting behind the prosody minimum.
    if (last.id !== lastMessageId.current) {
      lastMessageId.current = last.id;
      spokeThisTurn.current = false;
      // Whatever the previous message left unspoken — typically the sentence
      // said before a tool call — is still owed, but must not run into the
      // next message's first words.
      if (chunkBuffer.current && !chunkBuffer.current.endsWith("\n")) {
        chunkBuffer.current += "\n";
      }
    }

    const full = last.parts
      .map((part) => (part.type === "text" ? part.text : ""))
      .join("");
    const added = reader.current.read(last.id, full);
    chunkBuffer.current += added;

    // Only the first chunk races the clock; later ones read better long.
    const minChars = spokeThisTurn.current ? LATER_CHUNK_MIN_CHARS : 0;
    const { chunks, rest } = takeSpeakableChunks(
      chunkBuffer.current,
      false,
      minChars,
    );
    chunkBuffer.current = rest;
    chunks.forEach(speak);

    // Only new text restarts the wait; tool parts keep arriving and would
    // otherwise hold the pending sentence back for the whole tool chain.
    if (added) armTextSettledFlush();
  }

  /** Speak whatever is pending once the model stops adding to it. */
  function armTextSettledFlush() {
    setTimer("textSettled", TEXT_SETTLED_MS, () => {
      if (!chunkBuffer.current) return;
      const { chunks } = takeSpeakableChunks(chunkBuffer.current, true);
      chunkBuffer.current = "";
      chunks.forEach(speak);
    });
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

  function startTurn() {
    replyDone.current = false;
    spokeThisTurn.current = false;
    lastMessageId.current = "";
    chunkBuffer.current = "";
    reader.current.reset();
    clearTimer("session");
    dispatch({ type: "TRANSCRIPT_SENT" });
    armReplyWatchdog();
  }

  function armReplyWatchdog() {
    setTimer("reply", REPLY_SILENCE_MS, () => {
      replyDone.current = true;
      if (player().isIdle()) dispatch({ type: "REPLY_DONE" });
    });
  }

  function dispatch(event: VoiceEvent) {
    const next = voiceReduce(stateRef.current, event);
    if (next === stateRef.current) return;
    if (event.type === "REPLY_DONE") {
      trackVoiceMode("voice_turn_completed", { turn_index: turnIndex.current });
    }
    stateRef.current = next;
    setState(next);
    syncMic(next);
  }

  function syncMic(next: VoiceState) {
    if (isMicOpen(next)) vadRef.current?.resume();
    else vadRef.current?.pause();

    if (next === "listening") {
      setTimer("silence", inputs.current.silenceTimeoutMs, () =>
        deactivate("silence_timeout"),
      );
    } else {
      clearTimer("silence");
    }
    if (next === "listening" || next === "off") {
      clearTimer("reply");
      clearTimer("textSettled");
    }
  }

  function player(): SpeechPlayer {
    if (!playerRef.current) {
      playerRef.current = createSpeechPlayer({
        synthesize: (text) => synthesizeSpeech(text, inputs.current.sessionId),
        onIdle: () => {
          if (replyDone.current) dispatch({ type: "REPLY_DONE" });
        },
        onPlaybackStart: () => {
          if (!utteranceEndedAt.current) return;
          trackVoiceMode("voice_first_sound_latency_ms", {
            ms: Date.now() - utteranceEndedAt.current,
          });
          utteranceEndedAt.current = 0;
        },
        onError: (error) => {
          trackVoiceMode("voice_mode_error", { stage: "synthesis" });
          report(error);
        },
      });
    }
    return playerRef.current;
  }

  function teardown() {
    // Unmount without deactivate — navigating away mid-session. Leaving this
    // set would mark later text turns as voice turns.
    setVoiceTurnActive(false);
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
