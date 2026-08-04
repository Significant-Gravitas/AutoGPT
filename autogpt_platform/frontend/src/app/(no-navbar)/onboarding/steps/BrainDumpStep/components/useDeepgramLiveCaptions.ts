import { useEffect, useRef, useState } from "react";
import type { CaptionWord } from "./useLiveCaptions";

// Live captions via Deepgram streaming: the server mints a disposable
// access token, the browser streams raw PCM straight to Deepgram over a
// WebSocket and renders interim results word by word (~200-300ms behind
// speech). Purely cosmetic — the recording itself is still transcribed
// server-side after finalize.

const PCM_SAMPLE_RATE = 24000;
const VISIBLE_WORDS = 24;

const LISTEN_URL = `wss://api.deepgram.com/v1/listen?${new URLSearchParams({
  model: "nova-3",
  encoding: "linear16",
  sample_rate: String(PCM_SAMPLE_RATE),
  channels: "1",
  interim_results: "true",
  smart_format: "true",
})}`;

type LiveCaptionsStatus = "idle" | "connecting" | "live" | "failed";

interface DeepgramResult {
  type?: string;
  is_final?: boolean;
  channel?: { alternatives?: Array<{ transcript?: string }> };
}

export function useDeepgramLiveCaptions({
  enabled,
  audioStream,
}: {
  enabled: boolean;
  audioStream: MediaStream | null;
}) {
  const [words, setWords] = useState<CaptionWord[]>([]);
  const [status, setStatus] = useState<LiveCaptionsStatus>("idle");
  const wordsRef = useRef<CaptionWord[]>([]);
  const nextIdRef = useRef(0);

  useEffect(() => {
    if (!enabled || !audioStream) {
      setStatus("idle");
      return;
    }

    let disposed = false;
    let socket: WebSocket | null = null;
    let context: AudioContext | null = null;
    let processor: ScriptProcessorNode | null = null;
    let committedText = "";
    let partialText = "";

    setStatus("connecting");
    wordsRef.current = [];
    setWords([]);

    // Every exit runs the same teardown, so a failure stops the mic
    // reaching Deepgram instead of leaving an OPEN socket streaming
    // alongside whichever engine took over. Idempotent: the handlers are
    // detached before closing, so the resulting `onclose` cannot re-enter.
    function teardown() {
      processor?.disconnect();
      processor = null;
      void context?.close();
      context = null;
      if (!socket) return;
      socket.onerror = null;
      socket.onclose = null;
      socket.close();
      socket = null;
    }

    function fail() {
      teardown();
      if (!disposed) setStatus("failed");
    }

    function render() {
      const tokens = `${committedText} ${partialText}`.trim().split(/\s+/);
      // A word keeps the id of its slot when its text is revised, so the
      // marquee grows in place instead of remounting on every interim.
      const previous = wordsRef.current;
      const next = tokens.filter(Boolean).map((text, index) => {
        const slot = previous[index];
        if (!slot) return { id: nextIdRef.current++, text };
        return slot.text === text ? slot : { id: slot.id, text };
      });
      wordsRef.current = next;
      setWords(next.slice(-VISIBLE_WORDS));
    }

    function startAudio() {
      if (disposed || !audioStream) return;
      context = new AudioContext({ sampleRate: PCM_SAMPLE_RATE });
      const source = context.createMediaStreamSource(audioStream);
      // ScriptProcessor is deprecated but universally supported and its
      // zeroed output buffer means no mic echo on the destination.
      processor = context.createScriptProcessor(4096, 1, 1);
      source.connect(processor);
      processor.connect(context.destination);
      processor.onaudioprocess = (event) => {
        if (!socket || socket.readyState !== WebSocket.OPEN) return;
        socket.send(toPcm16(event.inputBuffer.getChannelData(0)));
      };
    }

    async function connect() {
      let accessToken: string;
      try {
        const response = await fetch("/api/transcribe/live-session", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ provider: "deepgram" }),
        });
        if (!response.ok) return fail();
        accessToken = (await response.json()).token;
      } catch {
        return fail();
      }
      if (disposed || !accessToken) return fail();

      // Browsers cannot set WebSocket headers; the disposable token rides
      // in a subprotocol instead. Nothing like the real API key.
      socket = new WebSocket(LISTEN_URL, ["bearer", accessToken]);
      socket.onopen = () => {
        if (disposed) return;
        // "live" only once the audio graph is actually running. If
        // AudioContext or the processor node throws, reporting live
        // would pin useLiveCaptions to a cloud engine that will never
        // send a word, instead of falling back to the browser one.
        try {
          startAudio();
        } catch {
          fail();
          return;
        }
        setStatus("live");
      };
      socket.onmessage = (message) => {
        const result = JSON.parse(message.data as string) as DeepgramResult;
        if (result.type !== "Results") return;
        const transcript = result.channel?.alternatives?.[0]?.transcript ?? "";
        // Interims keep replacing the tail until Deepgram finalises the
        // phrase, which commits it and starts a fresh tail.
        if (result.is_final) {
          if (transcript) committedText = `${committedText} ${transcript}`;
          partialText = "";
        } else {
          partialText = transcript;
        }
        render();
      };
      socket.onerror = fail;
      socket.onclose = fail;
    }

    void connect();

    return () => {
      disposed = true;
      teardown();
      wordsRef.current = [];
      setWords([]);
    };
  }, [enabled, audioStream]);

  return { words, status };
}

function toPcm16(samples: Float32Array): ArrayBuffer {
  const pcm = new Int16Array(samples.length);
  for (let index = 0; index < samples.length; index++) {
    const clamped = Math.max(-1, Math.min(1, samples[index]));
    pcm[index] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
  }
  return pcm.buffer;
}
