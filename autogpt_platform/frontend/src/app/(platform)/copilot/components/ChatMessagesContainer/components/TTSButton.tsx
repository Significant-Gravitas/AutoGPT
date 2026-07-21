"use client";

import { MessageAction } from "@/components/ai-elements/message";
import { SpeakerHigh, Stop } from "@phosphor-icons/react";
import { useTextToSpeech } from "@/components/contextual/Chat/components/ChatMessage/useTextToSpeech";
import { useMemo } from "react";

// Unicode emoji pattern (covers most emoji ranges including modifiers and ZWJ sequences)
const EMOJI_RE =
  /[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F1E0}-\u{1F1FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}\u{FE00}-\u{FE0F}\u{1F900}-\u{1F9FF}\u{1FA00}-\u{1FA6F}\u{1FA70}-\u{1FAFF}\u{200D}\u{20E3}\u{E0020}-\u{E007F}]+/gu;

function stripMarkdownForSpeech(md: string): string {
  return (
    md
      // Code blocks (``` ... ```)
      .replace(/```[\s\S]*?```/g, "")
      // Inline code
      .replace(/`([^`]*)`/g, "$1")
      // Images ![alt](url)
      .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
      // Links [text](url)
      .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
      // Bold/italic (***text***, **text**, *text*, ___text___, __text__, _text_)
      .replace(/\*{1,3}([^*]+)\*{1,3}/g, "$1")
      .replace(/_{1,3}([^_]+)_{1,3}/g, "$1")
      // Strikethrough
      .replace(/~~([^~]+)~~/g, "$1")
      // Headings (# ... ######)
      .replace(/^#{1,6}\s+/gm, "")
      // Horizontal rules
      .replace(/^[-*_]{3,}\s*$/gm, "")
      // Blockquotes
      .replace(/^>\s?/gm, "")
      // Unordered list markers
      .replace(/^[\s]*[-*+]\s+/gm, "")
      // Ordered list markers
      .replace(/^[\s]*\d+\.\s+/gm, "")
      // HTML tags
      .replace(/<[^>]+>/g, "")
      // Emoji
      .replace(EMOJI_RE, "")
      // Collapse multiple blank lines
      .replace(/\n{3,}/g, "\n\n")
      .trim()
  );
}

interface Props {
  text: string;
}

export function TTSButton({ text }: Props) {
  const cleanText = useMemo(() => stripMarkdownForSpeech(text), [text]);
  const { status, isSupported, toggle } = useTextToSpeech(cleanText);

  if (!isSupported || !cleanText) return null;

  const isPlaying = status === "playing";

  return (
    <MessageAction
      tooltip={isPlaying ? "Stop reading" : "Read aloud"}
      onClick={toggle}
    >
      {isPlaying ? <Stop size={16} /> : <SpeakerHigh size={16} />}
    </MessageAction>
  );
}
