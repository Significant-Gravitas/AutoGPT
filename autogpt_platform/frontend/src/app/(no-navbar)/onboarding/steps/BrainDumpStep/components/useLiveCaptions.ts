import { useEffect, useRef, useState } from "react";
import { useDeepgramLiveCaptions } from "./useDeepgramLiveCaptions";
import { useScribeLiveCaptions } from "./useScribeLiveCaptions";

// Cloud engines stream mic audio directly to the provider for accurate,
// fast captions; "browser" uses the free on-device SpeechRecognition.
// ElevenLabs Scribe is the default (best accuracy, ~150ms partials);
// "deepgram" is the A/B alternative (fast, looser on words). Either
// quietly falls back to the browser engine when its token can't be
// minted (no key) or the socket dies. Callers can override the engine
// per render (the brain-dump step's A/B tabs do); the env var sets the
// default.
export type CaptionsEngine = "browser" | "deepgram" | "elevenlabs";

function resolveDefaultEngine(): CaptionsEngine {
  const engine = process.env.NEXT_PUBLIC_LIVE_CAPTIONS_ENGINE;
  if (engine === "browser" || engine === "deepgram") return engine;
  return "elevenlabs";
}
export const DEFAULT_CAPTIONS_ENGINE = resolveDefaultEngine();

// Enough words to overflow the marquee, so the oldest ones are dropped
// well outside the box rather than popping out of view. This is a "we hear
// you" signal, not a transcript the user is meant to proofread.
const VISIBLE_WORDS = 24;

interface SpeechRecognitionLike {
  continuous: boolean;
  interimResults: boolean;
  start: () => void;
  stop: () => void;
  onresult: ((event: SpeechRecognitionResultEventLike) => void) | null;
  onerror: ((event?: { error?: string }) => void) | null;
  onend: (() => void) | null;
}

// The recogniser drops out constantly for reasons that resolve
// themselves — a pause in speech, a blip on the network — and `onend`
// just listens again. These are the ones that never recover, and after
// them the caption box would sit empty for the rest of the take unless
// the level meter takes over.
const FATAL_SPEECH_ERRORS = [
  "not-allowed",
  "service-not-allowed",
  "audio-capture",
  "language-not-supported",
];

interface SpeechRecognitionResultEventLike {
  results: ArrayLike<ArrayLike<{ transcript: string }>>;
}

type SpeechRecognitionConstructor = new () => SpeechRecognitionLike;

function getSpeechRecognition(): SpeechRecognitionConstructor | null {
  if (typeof window === "undefined") return null;
  const candidate = window as unknown as {
    SpeechRecognition?: SpeechRecognitionConstructor;
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
  };
  return (
    candidate.SpeechRecognition ?? candidate.webkitSpeechRecognition ?? null
  );
}

export interface CaptionWord {
  id: number;
  text: string;
}

export function useLiveCaptions({
  isRecording,
  audioStream,
  engine = DEFAULT_CAPTIONS_ENGINE,
}: {
  isRecording: boolean;
  audioStream: MediaStream | null;
  engine?: CaptionsEngine;
}) {
  const [words, setWords] = useState<CaptionWord[]>([]);
  const [level, setLevel] = useState(0);
  const [isSpeechSupported] = useState(() => getSpeechRecognition() !== null);
  // Feature detection only answers "is the API here", never "does it
  // work". On a Chrome that cannot reach the speech service, or a mic
  // revoked mid-take, the API is present and useless.
  const [speechFailed, setSpeechFailed] = useState(false);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  // Interim results rewrite the tail of an utterance, so a word keeps its
  // id while its text holds. Without stable ids the marquee would replay
  // its enter animation for every word on every partial result.
  const wordsRef = useRef<CaptionWord[]>([]);
  const nextIdRef = useRef(0);

  const scribe = useScribeLiveCaptions({
    enabled: engine === "elevenlabs" && isRecording,
    audioStream,
  });
  const deepgram = useDeepgramLiveCaptions({
    enabled: engine === "deepgram" && isRecording,
    audioStream,
  });
  const cloud = engine === "deepgram" ? deepgram : scribe;
  const useBrowserEngine = engine === "browser" || cloud.status === "failed";
  const canTranscribeLocally = isSpeechSupported && !speechFailed;

  useEffect(() => {
    if (!isRecording || !canTranscribeLocally || !useBrowserEngine) return;
    const Recognition = getSpeechRecognition();
    if (!Recognition) return;

    function replaceWords(next: CaptionWord[]) {
      wordsRef.current = next;
      setWords(next.slice(-VISIBLE_WORDS));
    }

    // The recogniser quietly ends itself all the time — after a few
    // seconds of silence, on a transient network error, after ~a minute
    // of continuous speech. Each session restarts from an empty result
    // list, so words committed before the restart are kept as a base that
    // new sessions append to.
    let disposed = false;
    let base: CaptionWord[] = [];
    let sessionStartedAt = 0;
    let rapidRestarts = 0;

    const recognition = new Recognition();
    recognition.continuous = true;
    recognition.interimResults = true;

    function begin() {
      sessionStartedAt = Date.now();
      try {
        recognition.start();
      } catch {
        // Already winding down from a previous session; the pending
        // `onend` will call begin() again.
      }
    }

    recognition.onresult = function handleResult(event) {
      rapidRestarts = 0;
      // Every utterance of this session, not just the latest one: the
      // line has to keep growing across utterance boundaries or it would
      // snap back to a single word each time a phrase is committed.
      const texts: string[] = [];
      for (let index = 0; index < event.results.length; index++) {
        const transcript = event.results[index]?.[0]?.transcript ?? "";
        for (const token of transcript.trim().split(/\s+/)) {
          if (token) texts.push(token);
        }
      }
      // A word keeps the id of its slot even when its text is revised, so
      // the word currently being recognised grows in place instead of
      // being torn down and re-mounted on every interim result.
      const previous = wordsRef.current;
      const session = texts.map((text, index) => {
        const slot = previous[base.length + index];
        if (!slot) return { id: nextIdRef.current++, text };
        return slot.text === text ? slot : { id: slot.id, text };
      });
      replaceWords([...base, ...session]);
    };
    // Recognition dropping out is cosmetic: the recording and the real
    // transcription are untouched, so the words already on screen stay
    // and `onend` decides whether to listen again. Only the errors it
    // can never come back from give up on the engine.
    recognition.onerror = function handleError(event) {
      if (FATAL_SPEECH_ERRORS.includes(event?.error ?? "")) {
        setSpeechFailed(true);
      }
    };
    recognition.onend = function handleEnd() {
      if (disposed) return;
      base = wordsRef.current;
      // A session dying immediately after starting means a fatal condition
      // (mic revoked, no speech service) — retrying forever would spin.
      if (Date.now() - sessionStartedAt < 1000) {
        rapidRestarts += 1;
        if (rapidRestarts > 5) {
          setSpeechFailed(true);
          return;
        }
      }
      begin();
    };
    begin();
    recognitionRef.current = recognition;

    return () => {
      disposed = true;
      recognition.stop();
      recognitionRef.current = null;
      replaceWords([]);
    };
  }, [isRecording, canTranscribeLocally, useBrowserEngine]);

  useEffect(() => {
    if (
      !isRecording ||
      !useBrowserEngine ||
      canTranscribeLocally ||
      !audioStream
    )
      return;
    const context = new AudioContext();
    const analyser = context.createAnalyser();
    analyser.fftSize = 256;
    context.createMediaStreamSource(audioStream).connect(analyser);
    const data = new Uint8Array(analyser.frequencyBinCount);

    let frame = 0;
    function sample() {
      analyser.getByteTimeDomainData(data);
      const peak = data.reduce(
        (max, value) => Math.max(max, Math.abs(value - 128)),
        0,
      );
      setLevel(Math.min(1, peak / 64));
      frame = requestAnimationFrame(sample);
    }
    sample();

    return () => {
      cancelAnimationFrame(frame);
      void context.close();
    };
  }, [isRecording, canTranscribeLocally, audioStream, useBrowserEngine]);

  return {
    words: useBrowserEngine ? words : cloud.words,
    level,
    // The cloud engines work everywhere a mic works, so "supported"
    // only depends on the browser API when we have fallen back to it —
    // and on whether that API is actually producing anything.
    isSpeechSupported: useBrowserEngine ? canTranscribeLocally : true,
  };
}
