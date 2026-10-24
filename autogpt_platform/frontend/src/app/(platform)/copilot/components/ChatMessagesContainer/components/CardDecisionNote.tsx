import { Cancel01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { type CardDecision, decisionNotes } from "../../../cardDecisions";
import { QueueBadge } from "./QueueBadge";

interface Props {
  messageID: string;
  decisions: CardDecision[];
  sessionID?: string | null;
  /** A shared chat: the note does not speak to the reader as "you". */
  readOnly: boolean;
  /** The decision is the latest message of a queued session: its turn waits
   *  for a free slot and can still be cancelled, as on any user row. */
  showQueueBadge: boolean;
}

/** A user row that only answered proposal cards, shown as what was decided.
 *  The row's text is the line Otto and the approval gate read. */
export function CardDecisionNote({
  messageID,
  decisions,
  sessionID,
  readOnly,
  showQueueBadge,
}: Props) {
  return (
    <div
      data-message-id={messageID}
      className="flex flex-col items-center gap-0.5 py-1"
    >
      {decisionNotes(decisions, { readOnly }).map((note, i) => (
        <Text
          key={i}
          variant="small"
          className="flex items-center gap-1.5 text-zinc-500"
        >
          <Icon
            icon={note.verdict === "approved" ? Tick02Icon : Cancel01Icon}
            size={14}
            className={
              note.verdict === "approved" ? "text-emerald-500" : "text-zinc-400"
            }
          />
          {note.text}
        </Text>
      ))}
      {showQueueBadge && (
        <div className="mt-1" data-testid="queue-status-row">
          <QueueBadge sessionID={sessionID ?? null} />
        </div>
      )}
    </div>
  );
}
