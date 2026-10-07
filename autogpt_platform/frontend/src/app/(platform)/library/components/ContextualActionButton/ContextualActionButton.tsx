"use client";
import { Button } from "@/components/atoms/Button/Button";
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

  function handleClick(e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();

    const params = new URLSearchParams();
    if (executionID) params.set("activeItem", executionID);
    const query = params.toString();
    router.push(`/library/agents/${agentID}${query ? `?${query}` : ""}`);
  }

  return (
    <Button
      type="button"
      variant="ghost"
      size="sm"
      onClick={handleClick}
      leftIcon={<Icon icon={config.icon} size={12} className="shrink-0" />}
      className={cn("text-zinc-600 hover:text-zinc-800", className)}
    >
      {config.label}
    </Button>
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
