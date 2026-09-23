"use client";

import { MessageAction } from "@/components/ai-elements/message";
import { StopIcon, VolumeHighIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { useTTSButton } from "./useTTSButton";

interface Props {
  text: string;
  sessionID: string | null;
}

export function TTSButton({ text, sessionID }: Props) {
  const { canSpeak, isPlaying, toggle } = useTTSButton({ text, sessionID });

  if (!canSpeak) return null;

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
