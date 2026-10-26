import { Cancel01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { useContext } from "react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  type CardDecision,
  decisionNotes,
  isRefusedAnswer,
} from "../../../cardDecisions";
import { AnsweredCardsContext } from "../../ToolChain/AnsweredCardsContext";
import { QueueBadge } from "./QueueBadge";

interface Props {
  messageID: string;
  decisions: CardDecision[];
  sessionID?: string | null;
  /** A shared chat: the note does not speak to the reader as "you". */
  readOnly: boolean;
  /** The row is saved; one only on screen can still be refused. */
  saved: boolean;
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
  saved,
  showQueueBadge,
}: Props) {
  const answered = useContext(AnsweredCardsContext);
  // A send the server refused stays on screen until a reload; the card and
  // the saved row already show the answer that stands.
  if (!saved && isRefusedAnswer(messageID, decisions, answered)) {
    return <div data-message-id={messageID} className="-mt-6 h-0" />;
  }
  return (
    <div
      data-message-id={messageID}
      className="flex flex-col items-center gap-0.5 py-1"
    >
      <DecisionNoteLines decisions={decisions} readOnly={readOnly} />
      {showQueueBadge && (
        <div className="mt-1" data-testid="queue-status-row">
          <QueueBadge sessionID={sessionID ?? null} />
        </div>
      )}
    </div>
  );
}

interface LinesProps {
  decisions: CardDecision[];
  readOnly: boolean;
}

export function DecisionNoteLines({ decisions, readOnly }: LinesProps) {
  return decisionNotes(decisions, { readOnly }).map((note, i) => (
    <Text
      key={i}
      variant="small"
      className="flex items-center gap-1.5 text-foreground-tertiary"
    >
      <Icon
        icon={note.verdict === "approved" ? Tick02Icon : Cancel01Icon}
        size={14}
        className={
          note.verdict === "approved"
            ? "text-success"
            : "text-foreground-subtle"
        }
      />
      {note.text}
    </Text>
  ));
}
