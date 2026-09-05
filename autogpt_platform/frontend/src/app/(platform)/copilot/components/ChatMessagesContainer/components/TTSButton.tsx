"use client";

import { MessageAction } from "@/components/ai-elements/message";
import { useTextToSpeech } from "@/components/contextual/Chat/components/ChatMessage/useTextToSpeech";
import { stripMarkdownForSpeech } from "../../../voice/stripMarkdownForSpeech";
import { useMemo } from "react";
import { StopIcon, VolumeHighIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
      {isPlaying ? (
        <Icon icon={StopIcon} size={16} />
      ) : (
        <Icon icon={VolumeHighIcon} size={16} />
      )}
    </MessageAction>
  );
}
