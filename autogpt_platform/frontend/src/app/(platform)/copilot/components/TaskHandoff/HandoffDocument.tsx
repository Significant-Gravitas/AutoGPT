"use client";

import { Text } from "@/components/atoms/Text/Text";

interface Props {
  title: string;
  /** What kind of file it is: "Markdown", "PDF". */
  kind: string;
  onOpen: () => void;
}

/** The full result, as a sheet lying in the bubble: a small page with its
 *  lines, its name, what it is, and one way in. */
export function HandoffDocument({ title, kind, onOpen }: Props) {
  return (
    <button
      type="button"
      onClick={onOpen}
      aria-label={`Read ${title}`}
      className="group/sheet flex w-full items-center gap-3 rounded-lg bg-surface py-2 pr-2.5 pl-2 text-left shadow-(--inbox-shadow-card) transition-shadow duration-150 ease-out outline-none hover:shadow-(--inbox-shadow-card-hover) focus-visible:ring-2 focus-visible:ring-focus-inbox"
    >
      <span
        aria-hidden
        className="flex h-11 w-9 shrink-0 flex-col gap-1 rounded-md bg-surface-tray px-1.5 pt-2 transition-transform duration-150 ease-out group-hover/sheet:-translate-y-0.5 group-hover/sheet:-rotate-3 motion-reduce:transition-none"
      >
        <span className="h-0.5 w-4 rounded-full bg-foreground-ink/40" />
        <span className="h-0.5 w-full rounded-full bg-foreground-ink/15" />
        <span className="h-0.5 w-full rounded-full bg-foreground-ink/15" />
        <span className="h-0.5 w-3 rounded-full bg-foreground-ink/15" />
      </span>
      <span className="flex min-w-0 flex-1 flex-col">
        <Text as="span" variant="inbox-medium" tone="ink" className="truncate">
          {title}
        </Text>
        <Text as="span" variant="small" tone="quiet">
          The full result · {kind}
        </Text>
      </span>
      <span className="flex h-7 shrink-0 items-center rounded-full bg-foreground-ink px-3 text-body-sm font-medium text-surface">
        Read
      </span>
    </button>
  );
}
