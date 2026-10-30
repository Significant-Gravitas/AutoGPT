import type { UIDataTypes, UIMessage, UITools } from "ai";
import { useState } from "react";
import { getLiveTurnEndAnnouncement } from "./helpers";

type ChatMessage = UIMessage<unknown, UIDataTypes, UITools>;

function replySignature(reply: ChatMessage | undefined) {
  if (!reply) return "";
  const last = reply.parts[reply.parts.length - 1];
  const state = last && "state" in last ? String(last.state) : "";
  const size = last && "text" in last ? last.text.length : 0;
  return `${reply.id}:${reply.parts.length}:${state}:${size}`;
}

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
  const [replyAtStart, setReplyAtStart] = useState(() =>
    replySignature(lastReply),
  );
  const [endedLive, setEndedLive] = useState(false);
  if (wasBusy !== isBusy) {
    setWasBusy(isBusy);
    if (isBusy) setReplyAtStart(replySignature(lastReply));
    // A send refused before the turn started leaves the earlier reply last:
    // nothing new finished, so nothing is announced. Compared by content,
    // since a rebuilt message list makes new objects for untouched replies.
    setEndedLive(
      wasBusy && !isBusy && replySignature(lastReply) !== replyAtStart,
    );
  }

  return (
    <div aria-live="polite" className="sr-only">
      {endedLive && isSettled ? getLiveTurnEndAnnouncement(messages) : ""}
    </div>
  );
}
