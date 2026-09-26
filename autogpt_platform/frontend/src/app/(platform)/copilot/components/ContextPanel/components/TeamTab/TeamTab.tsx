"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { UserGroupIcon } from "@hugeicons/core-free-icons";
import { DelegationRowView } from "./components/DelegationRowView";
import { useTeamRoster } from "./useTeamRoster";

interface Props {
  sessionId: string | null;
  withHeader?: boolean;
}

export function TeamTab({ sessionId, withHeader = false }: Props) {
  const { rows, liveCount, settledCount } = useTeamRoster(sessionId);

  if (rows.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 p-6 text-center">
        <Icon icon={UserGroupIcon} size={24} className="text-zinc-400" />
        <p className="text-sm font-medium text-zinc-700">
          No teammates at work
        </p>
        <p className="max-w-56 text-xs text-zinc-500">
          When AutoPilot delegates or hands off to an expert, they show up here
          with live status.
        </p>
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      {withHeader && (
        <div className="flex items-center gap-2 px-4 pb-1 pt-3">
          <span className="text-sm font-medium text-zinc-900">Team</span>
          {liveCount > 0 && (
            <span className="rounded-full bg-purple-50 px-1.5 text-[11px] font-medium text-purple-600">
              {liveCount} working
            </span>
          )}
        </div>
      )}
      <div className="flex min-h-0 flex-1 flex-col gap-1 overflow-y-auto px-2 py-2">
        {rows.map((row) => (
          <DelegationRowView key={row.key} row={row} />
        ))}
      </div>
      <footer className="flex items-center justify-between border-t border-zinc-200/70 px-4 py-1.5 text-[11px] text-zinc-500">
        <span className="flex items-center gap-2">
          {liveCount > 0 && (
            <span className="text-purple-600">● {liveCount} working</span>
          )}
          {settledCount > 0 && <span>{settledCount} settled</span>}
        </span>
        <span>
          {rows.length} {rows.length === 1 ? "teammate" : "teammates"}
        </span>
      </footer>
    </div>
  );
}
