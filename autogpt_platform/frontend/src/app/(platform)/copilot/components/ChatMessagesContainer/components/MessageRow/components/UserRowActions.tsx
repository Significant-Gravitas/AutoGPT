import { MessageActions } from "@/components/ai-elements/message";
import { CopyButton } from "../../CopyButton";
import { QueueBadge } from "../../QueueBadge";

interface Props {
  showQueueBadge: boolean;
  showCopy: boolean;
  createdAt?: string;
  copyText: string;
  sessionID?: string | null;
}

export function UserRowActions({
  showQueueBadge,
  showCopy,
  createdAt,
  copyText,
  sessionID,
}: Props) {
  const date = createdAt ? new Date(createdAt) : null;
  return (
    <>
      {showQueueBadge && (
        <MessageActions
          className="mt-1 items-center justify-end gap-1.5"
          data-testid="queue-status-row"
        >
          <QueueBadge sessionID={sessionID ?? null} />
        </MessageActions>
      )}
      {showCopy && (
        <MessageActions className="mt-1 items-center justify-end gap-2 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100 [@media(hover:none)]:opacity-100">
          {date && !Number.isNaN(date.getTime()) && (
            <span className="text-[11px] tabular-nums text-neutral-500 [@media(hover:none)]:hidden">
              {date.toLocaleString(undefined, {
                dateStyle: "medium",
                timeStyle: "short",
              })}
            </span>
          )}
          <CopyButton text={copyText} />
        </MessageActions>
      )}
    </>
  );
}
