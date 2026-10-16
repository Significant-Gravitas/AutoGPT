"use client";

import { Text } from "@/components/atoms/Text/Text";
import { formatTimeAgo } from "@/lib/utils/time";
import { cn } from "@/lib/utils";
import Link from "next/link";
import { ActorMark } from "../RecentWork/components/ActorMark";
import type { RecentWorkLine } from "./helpers";
import { useRecentWorkList } from "./useRecentWorkList";

export function RecentWorkList() {
  const { isLoading, lines } = useRecentWorkList();

  if (isLoading) return null;

  return (
    <section className="relative px-4 pb-10 text-left md:px-0">
      <Text
        variant="lead-semibold"
        as="h2"
        tone="primary"
        className="px-1 tracking-[-0.01em]"
      >
        Recent work
      </Text>
      <Text variant="body" tone="secondary" className="mt-0.5 px-1">
        What your team did lately.
      </Text>
      <ul className="mt-4 max-w-[48rem] divide-y divide-zinc-100 overflow-hidden rounded-xl border border-zinc-200/80 bg-white shadow-[0_1px_2px_rgba(0,0,0,0.04)]">
        {lines.map((line) => (
          <li key={line.id}>
            <RecentWorkRow line={line} />
          </li>
        ))}
        {lines.length === 0 && (
          <li className="px-4 py-6 text-center">
            <Text variant="body" tone="muted">
              Nothing yet. Files your team makes and work it finishes show up
              here.
            </Text>
          </li>
        )}
      </ul>
    </section>
  );
}

interface RowProps {
  line: RecentWorkLine;
}

function RecentWorkRow({ line }: RowProps) {
  const content = (
    <>
      <span className="shrink-0">
        <ActorMark
          actor={line.actor}
          size={22}
          className="size-[22px] rounded-full border border-zinc-200"
        />
      </span>
      <Text variant="body" tone="secondary" className="min-w-0 flex-1 truncate">
        <span className="font-medium text-zinc-900">{line.actor.name}</span>{" "}
        <span className={cn(line.failed && "text-red-600")}>{line.verb}</span>
        {line.object && (
          <>
            {" "}
            <span className="font-medium text-zinc-900">{line.object}</span>
          </>
        )}
        {line.suffix && <span className="text-zinc-500"> {line.suffix}</span>}
      </Text>
      <Text variant="small" tone="muted" className="shrink-0 tabular-nums">
        {formatTimeAgo(line.at)}
      </Text>
    </>
  );

  const rowClass = "flex items-center gap-3 px-4 py-3";
  if (!line.link) return <div className={rowClass}>{content}</div>;
  return (
    <Link
      href={line.link}
      className={cn(
        rowClass,
        "outline-none transition-colors hover:bg-zinc-50 focus-visible:bg-zinc-50",
      )}
    >
      {content}
    </Link>
  );
}
