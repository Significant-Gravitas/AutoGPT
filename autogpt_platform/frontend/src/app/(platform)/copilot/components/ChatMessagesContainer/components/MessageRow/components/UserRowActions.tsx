import { MessageActions } from "@/components/organisms/Message/Message";
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
        <MessageActions
          reveal="hover"
          className="mt-1 items-center justify-end gap-2"
        >
          {date && !Number.isNaN(date.getTime()) && (
            <span className="text-caption-sm leading-inherit text-foreground-tertiary tabular-nums [@media(hover:none)]:hidden">
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
