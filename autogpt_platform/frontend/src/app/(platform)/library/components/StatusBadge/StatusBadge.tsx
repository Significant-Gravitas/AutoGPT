"use client";

import { cn } from "@/lib/utils";
import type { AgentStatus } from "../../types";

const STATUS_CONFIG: Record<
  AgentStatus,
  { label: string; bg: string; text: string; pulse: boolean }
> = {
  running: {
    label: "Running",
    bg: "",
    text: "text-blue-600",
    pulse: true,
  },
  error: {
    label: "Error",
    bg: "",
    text: "text-red-500",
    pulse: false,
  },
  listening: {
    label: "Listening",
    bg: "",
    text: "text-purple-500",
    pulse: true,
  },
  scheduled: {
    label: "Scheduled",
    bg: "",
    text: "text-yellow-600",
    pulse: false,
  },
  idle: {
    label: "Idle",
    bg: "",
    text: "text-zinc-500",
    pulse: false,
  },
};

interface Props {
  status: AgentStatus;
  className?: string;
}

export function StatusBadge({ status, className }: Props) {
  const config = STATUS_CONFIG[status];

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium",
        config.bg,
        config.text,
        className,
      )}
    >
      <span
        className={cn(
          "inline-block h-1.5 w-1.5 rounded-full",
          config.pulse && "animate-pulse",
          statusDotColor(status),
        )}
      />
      {config.label}
    </span>
  );
}

function statusDotColor(status: AgentStatus): string {
  switch (status) {
    case "running":
      return "bg-blue-500";
    case "error":
      return "bg-red-500";
    case "listening":
      return "bg-purple-500";
    case "scheduled":
      return "bg-yellow-500";
    case "idle":
      return "bg-zinc-400";
  }
}
