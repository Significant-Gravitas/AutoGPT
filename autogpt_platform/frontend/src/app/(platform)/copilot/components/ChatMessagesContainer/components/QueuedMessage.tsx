import { Clock01Icon } from "@hugeicons/core-free-icons";
import {
  Message,
  MessageContent,
} from "@/components/organisms/Message/Message";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { getTextDecisions } from "../../../cardDecisions";
import { CredentialMentionText } from "../../CredentialMention/CredentialMentionText";
import { DecisionNoteLines } from "./CardDecisionNote";

interface Props {
  text: string;
  isCompact: boolean;
}

/** A message waiting for the running turn to end. One that only answers
 *  proposal cards shows what was decided, as it will once saved. */
export function QueuedMessage({ text, isCompact }: Props) {
  const decisions = getTextDecisions(text);
  if (decisions) {
    return (
      <div className="flex flex-col items-center gap-0.5 py-1">
        <DecisionNoteLines decisions={decisions} readOnly={false} />
        <QueuedLabel />
      </div>
    );
  }
  return (
    <Message from="user">
      <MessageContent
        className={cn(
          "flex flex-col gap-1 border border-dashed border-border-strong bg-surface-hover px-4 py-2.5 text-foreground-strong opacity-60",
          isCompact
            ? "rounded-xl text-sm leading-6"
            : "rounded-3xl text-body-lg leading-relaxed",
        )}
      >
        <span>
          <CredentialMentionText text={text} />
        </span>
        <QueuedLabel />
      </MessageContent>
    </Message>
  );
}

function QueuedLabel() {
  return (
    <span
      // TODO(tokens): no exact role for text-slate-500
      className="flex items-center gap-1 text-xs text-slate-500"
    >
      <Icon icon={Clock01Icon} className="size-3" />
      Queued
    </span>
  );
}
