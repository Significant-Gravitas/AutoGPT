"use client";

import { Button } from "@/components/atoms/Button/Button";
import {
  EyeIcon,
  ArrowsClockwiseIcon,
  MonitorPlayIcon,
  PlayIcon,
  ArrowCounterClockwiseIcon,
} from "@phosphor-icons/react";
import { useToast } from "@/components/molecules/Toast/use-toast";
import type { AgentStatus } from "../../types";

interface Props {
  status: AgentStatus;
  agentID: string;
  className?: string;
}

/**
 * Renders the single most relevant action for an agent based on its status.
 *
 * | Status    | Action          | Behaviour (TODO: wire to real endpoints) |
 * |-----------|-----------------|------------------------------------------|
 * | error     | View error      | Opens error detail / run log             |
 * | listening | Reconnect       | Opens reconnection flow                  |
 * | running   | Watch live      | Opens real-time execution view           |
 * | idle      | Run now         | Triggers immediate new run               |
 * | scheduled | Run now         | Triggers immediate new run               |
 */
export function ContextualActionButton({ status, agentID, className }: Props) {
  const { toast } = useToast();

  const config = ACTION_CONFIG[status];
  if (!config) return null;

  const Icon = config.icon;

  function handleClick(e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    // TODO: Replace with real API calls
    toast({
      title: config.label,
      description: `${config.label} triggered for agent ${agentID.slice(0, 8)}…`,
    });
  }

  return (
    <Button
      variant="outline"
      size="small"
      onClick={handleClick}
      leftIcon={<Icon size={14} />}
      className={className}
    >
      {config.label}
    </Button>
  );
}

const ACTION_CONFIG: Record<
  AgentStatus,
  { label: string; icon: typeof EyeIcon }
> = {
  error: { label: "View error", icon: EyeIcon },
  listening: { label: "Reconnect", icon: ArrowsClockwiseIcon },
  running: { label: "Watch live", icon: MonitorPlayIcon },
  idle: { label: "Run now", icon: PlayIcon },
  scheduled: { label: "Run now", icon: ArrowCounterClockwiseIcon },
};
