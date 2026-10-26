import type { UIDataTypes, UIMessage, UITools } from "ai";
import { useState } from "react";
import { getLiveTurnEndAnnouncement } from "./helpers";

interface Props {
  isStreaming: boolean;
  messages: UIMessage<unknown, UIDataTypes, UITools>[];
}

export function TurnEndAnnouncer({ isStreaming, messages }: Props) {
  const [wasStreaming, setWasStreaming] = useState(isStreaming);
  const [endedLive, setEndedLive] = useState(false);
  if (wasStreaming !== isStreaming) {
    setWasStreaming(isStreaming);
    setEndedLive(wasStreaming && !isStreaming);
  }

  return (
    <div aria-live="polite" className="sr-only">
      {endedLive ? getLiveTurnEndAnnouncement(messages) : ""}
    </div>
  );
}
