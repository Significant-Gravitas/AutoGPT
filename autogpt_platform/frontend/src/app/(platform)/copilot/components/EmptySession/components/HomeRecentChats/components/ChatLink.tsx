"use client";

import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { formatTimeAgo } from "@/lib/utils/time";
import Link from "next/link";
import { ChatStatusMark } from "./ChatStatusMark";

interface Props {
  session: SessionSummaryResponse;
  title: string;
  /** A task chat's row, a step shorter than its main chat's. */
  isNested?: boolean;
  className?: string;
}

/** One chat: its state, its title and when it last moved, opening it. */
export function ChatLink({ session, title, isNested, className }: Props) {
  return (
    <Link
      href={`/home?sessionId=${session.id}`}
      className={cn(
        "flex min-w-0 items-center gap-2.5 rounded-lg px-2.5 text-foreground-faint transition-colors hover:bg-surface-tray focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none",
        // Inset, since the branch clips what spills past it while it opens.
        isNested ? "h-8 focus-visible:ring-inset" : "h-9",
        className,
      )}
    >
      <ChatStatusMark session={session} />
      <Text
        variant="inbox"
        as="span"
        tone="ink"
        className="min-w-0 flex-1 truncate"
      >
        {title}
      </Text>
      <Text variant="inbox" as="span" tone="faint" className="shrink-0">
        {formatTimeAgo(session.updated_at)}
      </Text>
    </Link>
  );
}
