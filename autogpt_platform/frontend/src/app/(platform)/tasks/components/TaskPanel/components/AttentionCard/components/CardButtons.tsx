"use client";

import { Button } from "@/components/atoms/Button/Button";
import { cardActionLabel } from "../helpers";

interface Props {
  actions: string[];
  /** The work chat, where a file can be attached; null hides Upload file. */
  uploadHref: string | null;
  onAction(action: string): void;
  isActing: boolean;
  pendingAction?: string;
}

/** Retry first, Upload file next, then the other ways out. */
export function CardButtons({
  actions,
  uploadHref,
  onAction,
  isActing,
  pendingAction,
}: Props) {
  if (!actions.length && !uploadHref) return null;
  const [first, ...rest] = actions;

  return (
    <div className="flex flex-wrap gap-1.5">
      {first ? (
        <ActionButton
          action={first}
          primary
          onAction={onAction}
          isActing={isActing}
          pendingAction={pendingAction}
        />
      ) : null}
      {uploadHref ? (
        <Button
          as="NextLink"
          href={uploadHref}
          variant={first ? "secondary" : "primary"}
          size="xs"
          className="h-auto min-h-7 max-w-full min-w-0 text-left break-words whitespace-normal"
        >
          Upload file
        </Button>
      ) : null}
      {rest.map((action) => (
        <ActionButton
          key={action}
          action={action}
          onAction={onAction}
          isActing={isActing}
          pendingAction={pendingAction}
        />
      ))}
    </div>
  );
}

interface ActionButtonProps {
  action: string;
  primary?: boolean;
  onAction(action: string): void;
  isActing: boolean;
  pendingAction?: string;
}

function ActionButton({
  action,
  primary = false,
  onAction,
  isActing,
  pendingAction,
}: ActionButtonProps) {
  return (
    <Button
      variant={primary ? "primary" : "secondary"}
      size="xs"
      className="h-auto min-h-7 max-w-full min-w-0 text-left break-words whitespace-normal"
      disabled={isActing}
      loading={pendingAction === action}
      onClick={() => onAction(action)}
    >
      {cardActionLabel(action)}
    </Button>
  );
}
