import { useEffect, useRef, useState } from "react";
import type { CaptionWord } from "./useLiveCaptions";

// Live captions via ElevenLabs Scribe v2 Realtime: the server mints a
// single-use token, the browser streams PCM straight to ElevenLabs over a
// WebSocket and renders partial transcripts (~150ms behind speech).
// Purely cosmetic — the recording itself is still transcribed
// server-side after finalize.

const PCM_SAMPLE_RATE = 16000;
const VISIBLE_WORDS = 24;

function listenUrl(token: string) {
  const params = new URLSearchParams({
    model_id: "scribe_v2_realtime",
    audio_format: `pcm_${PCM_SAMPLE_RATE}`,
    commit_strategy: "vad",
    token,
  });
  return `wss://api.elevenlabs.io/v1/speech-to-text/realtime?${params}`;
}

type LiveCaptionsStatus = "idle" | "connecting" | "live" | "failed";

interface ScribeMessage {
  message_type?: string;
  text?: string;
  transcript?: string;
}

export function useScribeLiveCaptions({
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

    function fail() {
      if (!disposed) setStatus("failed");
    }

    function render() {
      const tokens = `${committedText} ${partialText}`.trim().split(/\s+/);
      // A word keeps the id of its slot when its text is revised, so the
      // marquee grows in place instead of remounting on every partial.
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
        socket.send(
          JSON.stringify({
            message_type: "input_audio_chunk",
            audio_base_64: pcm16Base64(event.inputBuffer.getChannelData(0)),
            sample_rate: PCM_SAMPLE_RATE,
          }),
        );
      };
    }

    async function connect() {
      let token: string;
      try {
        const response = await fetch("/api/transcribe/live-session", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ provider: "elevenlabs" }),
        });
        if (!response.ok) return fail();
        token = (await response.json()).token;
      } catch {
        return fail();
      }
      if (disposed || !token) return fail();

      // Single-use token rides in the query string — it is consumed on
      // connect and worthless afterwards. Nothing like the real API key.
      socket = new WebSocket(listenUrl(token));
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
        const event = JSON.parse(message.data as string) as ScribeMessage;
        const type = event.message_type ?? "";
        const text = event.text ?? event.transcript ?? "";
        // Partials keep replacing the tail until the VAD commits the
        // phrase, which locks it in and starts a fresh tail.
        if (type === "partial_transcript" || type === "final_transcript") {
          partialText = text;
          render();
        } else if (
          type === "committed_transcript" ||
          type === "committed_transcript_with_timestamps"
        ) {
          if (text) committedText = `${committedText} ${text}`;
          partialText = "";
          render();
        } else if (type.includes("error") || type === "quota_exceeded") {
          fail();
        }
      };
      socket.onerror = fail;
      socket.onclose = fail;
    }

    void connect();

    return () => {
      disposed = true;
      processor?.disconnect();
      void context?.close();
      socket?.close();
      wordsRef.current = [];
      setWords([]);
    };
  }, [enabled, audioStream]);

  return { words, status };
}

function pcm16Base64(samples: Float32Array): string {
  const pcm = new Int16Array(samples.length);
  for (let index = 0; index < samples.length; index++) {
    const clamped = Math.max(-1, Math.min(1, samples[index]));
    pcm[index] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
  }
  const bytes = new Uint8Array(pcm.buffer);
  let binary = "";
  for (let index = 0; index < bytes.length; index++) {
    binary += String.fromCharCode(bytes[index]);
  }
  return btoa(binary);
}
