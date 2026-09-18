"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import { LinkSquare01Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import {
  delegationActivityText,
  delegationStatusLabel,
  delegationTone,
  type DelegationKind,
  type DelegationTone,
  formatElapsed,
} from "../helpers";
import type { TeamRosterRow } from "../useTeamRoster";

interface Props {
  row: TeamRosterRow;
}

const TONE_DOT: Record<DelegationTone, string> = {
  working: "bg-purple-500",
  done: "bg-green-500",
  failed: "bg-red-500",
  stopped: "bg-zinc-400",
};

const KIND_LABEL: Record<DelegationKind, string> = {
  delegate: "Borrowed",
  handoff: "Handed over",
  sub_session: "Sub-AutoPilot",
};

export function DelegationRowView({ row }: Props) {
  const tone = delegationTone(row.effectiveStatus);
  const statusLabel = delegationStatusLabel(row.effectiveStatus);
  const activity = delegationActivityText(row, row.effectiveStatus);
  const name = row.expertName ?? "Sub-AutoPilot";
  const role =
    row.expertRole &&
    row.expertRole.trim().toLowerCase() !== name.trim().toLowerCase()
      ? row.expertRole
      : null;
  const meta = [
    KIND_LABEL[row.kind],
    row.runs > 1 ? `run ${row.runs}` : null,
    row.elapsedSeconds !== null ? formatElapsed(row.elapsedSeconds) : null,
  ].filter((value): value is string => value !== null);

  return (
    <div
      data-testid="delegation-row"
      className="grid h-[3.875rem] grid-cols-[1.25rem_minmax(0,1fr)_auto] grid-rows-[1.25rem_1.125rem_1rem] items-center gap-x-2 rounded-xl px-2 py-1 transition-colors hover:bg-white/70"
    >
      <span className="col-start-1 row-start-1 row-end-3 flex items-start pt-0.5">
        <ExpertAvatar name={name} avatarUrl={row.expertAvatarUrl} size={20} />
      </span>
      <span className="col-start-2 row-start-1 flex min-w-0 items-baseline gap-1.5">
        <span className="min-w-0 truncate text-sm font-medium text-zinc-800">
          {name}
        </span>
        {role && (
          <span className="max-w-28 shrink-0 truncate rounded border border-zinc-200 px-1 text-[10px] text-zinc-500">
            {role}
          </span>
        )}
      </span>
      <span className="col-start-3 row-start-1 flex items-center gap-1.5 text-[11px] text-zinc-500">
        <span
          aria-hidden
          className={cn(
            "size-1.5 shrink-0 rounded-full",
            TONE_DOT[tone],
            tone === "working" && "animate-pulse motion-reduce:animate-none",
          )}
        />
        <span>{statusLabel}</span>
      </span>
      <span
        className={cn(
          "col-start-2 col-end-4 row-start-2 block truncate text-xs",
          tone === "failed" ? "text-red-500" : "text-zinc-500",
        )}
        title={activity ?? undefined}
      >
        {activity ?? statusLabel}
      </span>
      <span className="col-start-2 col-end-4 row-start-3 flex min-w-0 items-center gap-1 text-[10px] text-zinc-400">
        <span className="min-w-0 truncate">{meta.join(" · ")}</span>
        {row.link && (
          <Link
            href={row.link}
            aria-label={`Open ${name}'s session`}
            className="ml-auto shrink-0 rounded-full p-0.5 text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-zinc-700"
          >
            <Icon icon={LinkSquare01Icon} size={12} />
          </Link>
        )}
      </span>
    </div>
  );
}
