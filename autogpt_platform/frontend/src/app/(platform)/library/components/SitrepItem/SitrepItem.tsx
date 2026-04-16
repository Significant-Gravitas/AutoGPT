"use client";

import { Text } from "@/components/atoms/Text/Text";
import {
  WarningCircleIcon,
  ClockCountdownIcon,
  CheckCircleIcon,
  ChatCircleDotsIcon,
  EarIcon,
  CalendarDotsIcon,
  MoonIcon,
  EyeIcon,
} from "@phosphor-icons/react";
import NextLink from "next/link";
import { cn } from "@/lib/utils";
import { useRouter } from "next/navigation";
import type { SitrepItemData, SitrepPriority } from "../../types";
import { ContextualActionButton } from "../ContextualActionButton/ContextualActionButton";
import styles from "./SitrepItem.module.css";

interface Props {
  item: SitrepItemData;
}

const PRIORITY_CONFIG: Record<
  SitrepPriority,
  {
    icon?: typeof WarningCircleIcon;
    color: string;
    bg: string;
    cssSpinner?: boolean;
  }
> = {
  error: {
    icon: WarningCircleIcon,
    color: "text-red-500",
    bg: "bg-red-50",
  },
  running: {
    color: "text-zinc-800",
    bg: "",
    cssSpinner: true,
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
  listening: {
    icon: EarIcon,
    color: "text-purple-500",
    bg: "bg-purple-50",
  },
  scheduled: {
    icon: CalendarDotsIcon,
    color: "text-yellow-600",
    bg: "bg-yellow-50",
  },
  idle: {
    icon: MoonIcon,
    color: "text-zinc-400",
    bg: "bg-zinc-100",
  },
};

export function SitrepItem({ item }: Props) {
  const config = PRIORITY_CONFIG[item.priority];
  const router = useRouter();

  function handleAskAutoPilot() {
    const prompt = buildAutoPilotPrompt(item);
    const encoded = encodeURIComponent(prompt);
    router.push(`/copilot?autosubmit=true#prompt=${encoded}`);
  }

  return (
    <div
      className={cn(
        "flex flex-col gap-2 rounded-medium border border-zinc-200/50 bg-transparent p-2 sm:flex-row sm:items-center sm:gap-3",
      )}
    >
      <div className="flex min-w-0 flex-1 items-center gap-3">
        {item.agentImageUrl ? (
          <img
            src={item.agentImageUrl}
            alt={item.agentName}
            className="h-6 w-6 flex-shrink-0 rounded-full object-cover"
          />
        ) : (
          <div
            className={cn(
              "flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-full",
              config.bg,
            )}
          >
            {config.cssSpinner ? (
              <div
                className={cn(
                  styles.spinner,
                  "h-[21px] w-[21px] text-zinc-800",
                )}
              />
            ) : (
              config.icon && (
                <config.icon size={14} className={config.color} weight="fill" />
              )
            )}
          </div>
        )}

        <div className="min-w-0 flex-1">
          <Text variant="body-medium" className="leading-tight text-zinc-900">
            {item.agentName}
          </Text>
          <Text variant="small" className="leading-tight text-zinc-500">
            {item.message}
          </Text>
        </div>
      </div>

      <div className="flex flex-shrink-0 flex-wrap items-center justify-center gap-1.5 sm:flex-nowrap sm:justify-end">
        {item.priority === "success" ? (
          <NextLink
            href={`/library/agents/${item.agentID}${item.executionID ? `?activeItem=${item.executionID}` : ""}`}
            className="inline-flex items-center gap-1 rounded-md px-2 py-1.5 text-[13px] font-medium text-zinc-600 transition-colors hover:bg-zinc-50 hover:text-zinc-800"
          >
            <EyeIcon size={14} className="shrink-0" />
            See task
          </NextLink>
        ) : (
          <ContextualActionButton
            status={item.status}
            agentID={item.agentID}
            executionID={item.executionID}
          />
        )}
        <button
          type="button"
          onClick={handleAskAutoPilot}
          className="inline-flex items-center gap-1 rounded-md px-2 py-1.5 text-[13px] font-medium text-zinc-600 transition-colors hover:bg-zinc-50 hover:text-zinc-800"
        >
          <ChatCircleDotsIcon size={14} className="shrink-0" />
          Ask AutoPilot
        </button>
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
      return `Show me what ${item.agentName} found in its last run — summarize the results and any key takeaways.`;
    case "listening":
      return `What is ${item.agentName} listening for? Give me a summary of its trigger configuration.`;
    case "scheduled":
      return `When is ${item.agentName} scheduled to run next?`;
    case "idle":
      return `${item.agentName} has been idle. Should I keep it or update and re-run it?`;
  }
}
