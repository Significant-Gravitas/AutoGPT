import NextLink from "next/link";
import type { SitrepItemData } from "@/app/(platform)/library/types";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { buildAskHref, getRunStatus } from "../runHelpers";

interface Props {
  run: SitrepItemData;
}

const ACTION_CLASS =
  "rounded px-1.5 py-0.5 text-[11px] font-medium text-zinc-500 outline-none transition-colors hover:bg-zinc-100 hover:text-zinc-900 focus-visible:bg-zinc-100";

export function RunRow({ run }: Props) {
  const status = getRunStatus(run);
  return (
    <div className="group flex items-center gap-3 px-4 py-2 transition-colors hover:bg-zinc-50">
      <span className="flex w-24 shrink-0 items-center gap-1.5 text-sm font-medium text-zinc-500">
        <span
          className={cn(
            "size-1.5 rounded-full",
            status.dot,
            status.pulse && "animate-pulse",
          )}
          aria-hidden="true"
        />
        {status.label}
      </span>
      <div className="flex min-w-0 flex-1 items-baseline gap-2">
        <Text variant="body-medium" className="shrink-0 truncate text-zinc-900">
          {run.agentName}
        </Text>
        <Text variant="body" className="min-w-0 truncate text-zinc-500">
          {run.message}
        </Text>
      </div>
      {/* The actions ride on hover and focus so the rows stay quiet; they
          remain in the accessibility tree either way. */}
      <div className="flex shrink-0 items-center gap-0.5 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100">
        <NextLink
          href={`/library/agents/${run.agentID}`}
          className={ACTION_CLASS}
        >
          See
        </NextLink>
        <NextLink href={buildAskHref(run)} className={ACTION_CLASS}>
          Ask
        </NextLink>
      </div>
    </div>
  );
}
