"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { useTextToSpeech } from "@/components/contextual/Chat/components/ChatMessage/useTextToSpeech";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";

import { synthesizeSpeech } from "../../../voice/speechApi";
import { takeSpeakableChunks } from "../../../voice/speechChunker";
import {
  createSpeechPlayer,
  type SpeechPlayer,
} from "../../../voice/speechPlayer";
import { stripMarkdownForSpeech } from "../../../voice/stripMarkdownForSpeech";

interface Args {
  text: string;
  sessionID: string | null;
}

export function useTTSButton({ text, sessionID }: Args) {
  const cleanText = useMemo(() => stripMarkdownForSpeech(text), [text]);
  const browser = useTextToSpeech(cleanText);
  const server = useServerSpeech(cleanText, sessionID);

  // The speech route 404s when voice mode is off for the user, so without the
  // flag there is nothing to fall back to and the button must not offer one.
  const canFallBack = useGetFlag(Flag.COPILOT_VOICE_MODE);
  const viaServer = !browser.hasVoices;

  return {
    canSpeak: Boolean(cleanText) && (browser.hasVoices || canFallBack),
    isPlaying: viaServer ? server.isPlaying : browser.status === "playing",
    toggle: viaServer ? server.toggle : browser.toggle,
  };
}

function useServerSpeech(text: string, sessionID: string | null) {
  const { toast } = useToast();
  const [isPlaying, setIsPlaying] = useState(false);
  const playerRef = useRef<SpeechPlayer | null>(null);
  const reportedRef = useRef(false);
  const sessionRef = useRef(sessionID);
  sessionRef.current = sessionID;

  useEffect(() => {
    playerRef.current?.stop();
    setIsPlaying(false);
  }, [text]);

  useEffect(() => () => playerRef.current?.destroy(), []);

  return { isPlaying, toggle };

  function toggle() {
    if (isPlaying) {
      playerRef.current?.stop();
      setIsPlaying(false);
      return;
    }

    const player = ensurePlayer();
    // Must happen inside the click: that is the only gesture granting the
    // shared <audio> element autoplay for the chunks that arrive later.
    player.unlock();
    reportedRef.current = false;
    setIsPlaying(true);
    // One request is capped at 4096 characters, which a long reply exceeds.
    takeSpeakableChunks(text, true).chunks.forEach((chunk) =>
      player.enqueue(chunk),
    );
  }

  function ensurePlayer(): SpeechPlayer {
    if (!playerRef.current) {
      playerRef.current = createSpeechPlayer({
        synthesize: (chunk, kind) =>
          synthesizeSpeech(chunk, sessionRef.current, kind),
        onIdle: () => setIsPlaying(false),
        onError: report,
      });
    }
    return playerRef.current;
  }

  // Every remaining chunk of a failed reply fails the same way; say it once.
  function report(error: unknown) {
    if (reportedRef.current) return;
    reportedRef.current = true;
    playerRef.current?.stop();
    setIsPlaying(false);
    console.error("[Read aloud]", error);
    toast({
      title: "Read aloud",
      description:
        error instanceof Error ? error.message : "Something went wrong.",
      variant: "destructive",
    });
  }
}
