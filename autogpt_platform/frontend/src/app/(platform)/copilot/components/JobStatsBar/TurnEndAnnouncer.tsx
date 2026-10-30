import type { UIDataTypes, UIMessage, UITools } from "ai";
import { useState } from "react";
import { getLiveTurnEndAnnouncement } from "./helpers";

type ChatMessage = UIMessage<unknown, UIDataTypes, UITools>;

interface Props {
  isBusy: boolean;
  isSettled: boolean;
  messages: ChatMessage[];
}

export function TurnEndAnnouncer({ isBusy, isSettled, messages }: Props) {
  const lastReply = messages.findLast(
    (message) => message.role === "assistant",
  );
  const [wasBusy, setWasBusy] = useState(isBusy);
  const [replyAtStart, setReplyAtStart] = useState(lastReply);
  const [endedLive, setEndedLive] = useState(false);
  if (wasBusy !== isBusy) {
    setWasBusy(isBusy);
    if (isBusy) setReplyAtStart(lastReply);
    // A send refused before the turn started leaves the earlier reply last,
    // untouched: nothing new finished, so nothing is announced.
    setEndedLive(wasBusy && !isBusy && lastReply !== replyAtStart);
  }

  return (
    <div aria-live="polite" className="sr-only">
      {endedLive && isSettled ? getLiveTurnEndAnnouncement(messages) : ""}
    </div>
  );
}
