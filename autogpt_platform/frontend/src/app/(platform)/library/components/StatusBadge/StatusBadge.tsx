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
    text: "text-info-foreground",
    pulse: true,
  },
  error: {
    label: "Error",
    bg: "",
    text: "text-error",
    pulse: false,
  },
  listening: {
    label: "Listening",
    bg: "",
    text: "text-brand",
    pulse: true,
  },
  scheduled: {
    label: "Scheduled",
    bg: "",
    // oxlint-disable-next-line design-system/no-raw-colors -- yellow-600 #DFAF00 has no role; gold to amber change deferred (see PR)
    text: "text-yellow-600",
    pulse: false,
  },
  idle: {
    label: "Idle",
    bg: "",
    text: "text-foreground-tertiary",
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
      return "bg-info";
    case "error":
      return "bg-error";
    case "listening":
      return "bg-brand";
    case "scheduled":
      // oxlint-disable-next-line design-system/no-raw-colors -- yellow-500 #F5C000 has no role; yellow to amber change deferred (see PR)
      return "bg-yellow-500";
    case "idle":
      return "bg-foreground-subtle";
  }
}
