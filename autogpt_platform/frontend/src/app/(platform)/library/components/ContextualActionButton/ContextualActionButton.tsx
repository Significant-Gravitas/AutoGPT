"use client";
import { cn } from "@/lib/utils";
import { useRouter } from "next/navigation";
import type { AgentStatus } from "../../types";
import {
  ComputerVideoIcon,
  EyeIcon,
  ReloadIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { getLibraryAgentHref } from "@/services/org-team/builder";

interface Props {
  status: AgentStatus;
  agentID: string;
  executionID?: string;
  organizationID: string | null;
  teamID: string | null;
  className?: string;
}

export function ContextualActionButton({
  status,
  agentID,
  executionID,
  organizationID,
  teamID,
  className,
}: Props) {
  const router = useRouter();

  const config = ACTION_CONFIG[status];
  if (!config) return null;

  function handleClick(e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();

    router.push(
      getLibraryAgentHref(
        agentID,
        organizationID,
        teamID,
        executionID,
        executionID ? "runs" : null,
      ),
    );
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
      <Icon icon={config.icon} size={12} className="shrink-0" />
      {config.label}
    </button>
  );
}

const ACTION_CONFIG: Record<
  AgentStatus,
  { label: string; icon: IconSvgElement }
> = {
  error: { label: "View error", icon: EyeIcon },
  listening: { label: "Reconnect", icon: ReloadIcon },
  running: { label: "Watch live", icon: ComputerVideoIcon },
  idle: { label: "View", icon: EyeIcon },
  scheduled: { label: "View", icon: EyeIcon },
};
