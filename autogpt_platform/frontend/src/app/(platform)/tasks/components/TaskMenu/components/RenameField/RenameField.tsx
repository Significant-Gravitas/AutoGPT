"use client";

import { cn } from "@/lib/utils";
import { useRenameField } from "./useRenameField";

interface Props {
  taskId: string;
  title: string;
  onDone: () => void;
  className?: string;
}

/** The task's title, editable in place: Enter or leaving the field saves,
 *  Escape keeps the old title. */
export function RenameField({ taskId, title, onDone, className }: Props) {
  const field = useRenameField({ taskId, title, onDone });

  return (
    <input
      aria-label="Task title"
      value={field.text}
      onChange={(e) => field.setText(e.target.value)}
      onKeyDown={field.onKeyDown}
      onBlur={field.save}
      autoFocus
      maxLength={200}
      className={cn(
        "w-full min-w-0 rounded-control bg-surface px-2 py-1 text-body-sm font-medium text-foreground-ink shadow-pill focus:ring-2 focus:ring-focus-inbox focus:outline-none",
        className,
      )}
    />
  );
}
