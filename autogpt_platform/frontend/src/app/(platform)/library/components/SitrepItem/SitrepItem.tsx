"use client";

import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import {
  WarningCircleIcon,
  PlayIcon,
  ClockCountdownIcon,
  CheckCircleIcon,
  ChatCircleDotsIcon,
} from "@phosphor-icons/react";
import { cn } from "@/lib/utils";
import type { AgentStatus } from "../../types";
import { ContextualActionButton } from "../ContextualActionButton/ContextualActionButton";

export type SitrepPriority = "error" | "running" | "stale" | "success";

export interface SitrepItemData {
  id: string;
  agentID: string;
  agentName: string;
  priority: SitrepPriority;
  message: string;
  status: AgentStatus;
}

interface Props {
  item: SitrepItemData;
  onAskAutoPilot?: (prompt: string) => void;
}

const PRIORITY_CONFIG: Record<
  SitrepPriority,
  { icon: typeof WarningCircleIcon; color: string; bg: string }
> = {
  error: {
    icon: WarningCircleIcon,
    color: "text-red-500",
    bg: "bg-red-50",
  },
  running: {
    icon: PlayIcon,
    color: "text-blue-600",
    bg: "bg-blue-50",
  },
  stale: {
    icon: ClockCountdownIcon,
    color: "text-yellow-600",
    bg: "bg-yellow-50",
  },
  success: {
    icon: CheckCircleIcon,
    color: "text-green-600",
    bg: "bg-green-50",
  },
};

export function SitrepItem({ item, onAskAutoPilot }: Props) {
  const config = PRIORITY_CONFIG[item.priority];
  const Icon = config.icon;

  function handleAskAutoPilot() {
    const prompt = buildAutoPilotPrompt(item);
    onAskAutoPilot?.(prompt);
  }

  return (
    <div
      className={cn(
        "group flex items-start gap-3 rounded-medium border border-transparent p-3 transition-colors hover:border-zinc-100 hover:bg-zinc-50/50",
      )}
    >
      <div
        className={cn(
          "mt-0.5 flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-full",
          config.bg,
        )}
      >
        <Icon size={14} className={config.color} weight="fill" />
      </div>

      <div className="min-w-0 flex-1">
        <Text variant="small-medium" className="text-zinc-900">
          {item.agentName}
        </Text>
        <Text variant="small" className="mt-0.5 text-zinc-500">
          {item.message}
        </Text>
      </div>

      <div className="flex flex-shrink-0 items-center gap-1.5 opacity-0 transition-opacity group-hover:opacity-100">
        <ContextualActionButton status={item.status} agentID={item.agentID} />
        <Button
          variant="ghost"
          size="small"
          onClick={handleAskAutoPilot}
          leftIcon={<ChatCircleDotsIcon size={14} />}
          className="text-xs"
        >
          Ask AutoPilot
        </Button>
      </div>
    </div>
  );
}

function buildAutoPilotPrompt(item: SitrepItemData): string {
  switch (item.priority) {
    case "error":
      return `What happened with ${item.agentName}? It says "${item.message}" — can you check the logs and tell me what to fix?`;
    case "running":
      return `Give me a status update on the ${item.agentName} run — what has it found so far?`;
    case "stale":
      return `${item.agentName} hasn't run recently. Should I keep it or update and re-run it?`;
    case "success":
      return `How has ${item.agentName} been performing? Give me a quick summary of recent results.`;
  }
}
