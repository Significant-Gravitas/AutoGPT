"use client";

import {
  EyeIcon,
  ArrowsClockwiseIcon,
  MonitorPlayIcon,
  PlayIcon,
  ArrowCounterClockwiseIcon,
} from "@phosphor-icons/react";
import { cn } from "@/lib/utils";
import { useRouter } from "next/navigation";
import type { AgentStatus } from "../../types";

interface Props {
  status: AgentStatus;
  agentID: string;
  executionID?: string;
  className?: string;
}

export function ContextualActionButton({
  status,
  agentID,
  executionID,
  className,
}: Props) {
  const router = useRouter();

  const config = ACTION_CONFIG[status];
  if (!config) return null;

  const Icon = config.icon;

  function handleClick(e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();

    const params = new URLSearchParams();
    if (executionID) params.set("activeItem", executionID);
    const query = params.toString();
    router.push(`/library/agents/${agentID}${query ? `?${query}` : ""}`);
  }

  return (
    <button
      type="button"
      onClick={handleClick}
      className={cn(
        "inline-flex items-center gap-1 rounded-md px-2 py-1.5 text-[13px] font-medium text-zinc-600 transition-colors hover:bg-zinc-50 hover:text-zinc-800",
        className,
      )}
    >
      <Icon size={12} className="shrink-0" />
      {config.label}
    </button>
  );
}

const ACTION_CONFIG: Record<
  AgentStatus,
  { label: string; icon: typeof EyeIcon }
> = {
  error: { label: "View error", icon: EyeIcon },
  listening: { label: "Reconnect", icon: ArrowsClockwiseIcon },
  running: { label: "Watch live", icon: MonitorPlayIcon },
  idle: { label: "Start", icon: PlayIcon },
  scheduled: { label: "Start", icon: ArrowCounterClockwiseIcon },
};
