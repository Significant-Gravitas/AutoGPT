"use client";

import type { CoPilotUsageStatus } from "@/app/api/__generated__/models/coPilotUsageStatus";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useGetV2GetCopilotUsage } from "@/app/api/__generated__/endpoints/chat/chat";
import {
  formatResetTime,
  formatCents,
} from "@/app/(platform)/copilot/components/usageHelpers";
import { useResetRateLimit } from "@/app/(platform)/copilot/hooks/useResetRateLimit";
import { Button } from "@/components/atoms/Button/Button";
import { Badge } from "@/components/atoms/Badge/Badge";
import useCredits from "@/hooks/useCredits";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useSitrepItems } from "../SitrepItem/useSitrepItems";
import { SitrepItem } from "../SitrepItem/SitrepItem";
import { useAgentStatusMap } from "../../hooks/useAgentStatus";
import type { AgentStatusFilter } from "../../types";
import { Text } from "@/components/atoms/Text/Text";
import Link from "next/link";
import { useState } from "react";

interface Props {
  activeTab: AgentStatusFilter;
  agents: LibraryAgent[];
}

export function BriefingTabContent({ activeTab, agents }: Props) {
  if (activeTab === "all") {
    return <UsageSection />;
  }

  if (
    activeTab === "running" ||
    activeTab === "attention" ||
    activeTab === "completed"
  ) {
    return <ExecutionListSection activeTab={activeTab} agents={agents} />;
  }

  return <AgentListSection activeTab={activeTab} agents={agents} />;
}

function UsageSection() {
  const { data: usage } = useGetV2GetCopilotUsage({
    query: {
      select: (res) => res.data as CoPilotUsageStatus,
      refetchInterval: 30000,
      staleTime: 10000,
    },
  });

  const isBillingEnabled = useGetFlag(Flag.ENABLE_PLATFORM_PAYMENT);
  const { credits, fetchCredits } = useCredits({ fetchInitialCredits: true });
  const resetCost = usage?.reset_cost;
  const hasInsufficientCredits =
    credits !== null && resetCost != null && credits < resetCost;

  if (!usage?.daily || !usage?.weekly) return null;

  return (
    <div className="py-2">
      <div className="flex items-center gap-2">
        <Text variant="h5" className="text-neutral-800">
          Usage limits
        </Text>
        {usage.tier && (
          <Badge variant="info" size="small" className="bg-[rgb(224,237,255)]">
            {usage.tier.charAt(0) + usage.tier.slice(1).toLowerCase()} plan
          </Badge>
        )}
        <div className="flex-1" />
        {isBillingEnabled && (
          <Link
            href="/profile/credits"
            className="text-sm text-blue-600 hover:underline"
          >
            Manage billing
          </Link>
        )}
      </div>
      <div className="mt-4 grid grid-cols-1 gap-6 sm:grid-cols-2">
        {usage.daily.limit > 0 && (
          <UsageMeter
            label="Today"
            used={usage.daily.used}
            limit={usage.daily.limit}
            resetsAt={usage.daily.resets_at}
          />
        )}
        {usage.weekly.limit > 0 && (
          <UsageMeter
            label="This week"
            used={usage.weekly.used}
            limit={usage.weekly.limit}
            resetsAt={usage.weekly.resets_at}
          />
        )}
      </div>
      <UsageFooter
        usage={usage}
        hasInsufficientCredits={hasInsufficientCredits}
        onCreditChange={fetchCredits}
      />
    </div>
  );
}

const MAX_VISIBLE = 6;

function ExecutionListSection({
  activeTab,
  agents,
}: {
  activeTab: AgentStatusFilter;
  agents: LibraryAgent[];
}) {
  const allItems = useSitrepItems(agents, 50);
  const [showAll, setShowAll] = useState(false);

  const filtered = allItems.filter((item) => {
    if (activeTab === "running") return item.priority === "running";
    if (activeTab === "attention") return item.priority === "error";
    if (activeTab === "completed") return item.priority === "success";
    return false;
  });

  if (filtered.length === 0) {
    return <EmptyMessage tab={activeTab} />;
  }

  const visible = showAll ? filtered : filtered.slice(0, MAX_VISIBLE);
  const hasMore = filtered.length > MAX_VISIBLE;

  return (
    <div>
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        {visible.map((item) => (
          <SitrepItem key={item.id} item={item} />
        ))}
      </div>
      {hasMore && (
        <div className="mt-3 flex justify-center">
          <Button
            variant="secondary"
            size="small"
            onClick={() => setShowAll(!showAll)}
          >
            {showAll ? "Collapse" : `Show all (${filtered.length})`}
          </Button>
        </div>
      )}
    </div>
  );
}

const TAB_STATUS_LABEL: Record<string, string> = {
  listening: "Waiting for trigger event",
  scheduled: "Has a scheduled run",
  idle: "No recent activity",
};

function getAgentStatusLabel(tab: string, agent: LibraryAgent): string {
  if (tab === "scheduled" && agent.next_scheduled_run) {
    const diff = new Date(agent.next_scheduled_run).getTime() - Date.now();
    const minutes = Math.round(diff / 60_000);
    if (minutes <= 0) return "Scheduled to run soon";
    if (minutes < 60) return `Scheduled to run in ${minutes}m`;
    const hours = Math.round(minutes / 60);
    if (hours < 24) return `Scheduled to run in ${hours}h`;
    const days = Math.round(hours / 24);
    return `Scheduled to run in ${days}d`;
  }
  return TAB_STATUS_LABEL[tab] ?? "";
}

function AgentListSection({
  activeTab,
  agents,
}: {
  activeTab: AgentStatusFilter;
  agents: LibraryAgent[];
}) {
  const [showAll, setShowAll] = useState(false);
  const statusMap = useAgentStatusMap(agents);

  const filtered = agents.filter((agent) => {
    const status = statusMap.get(agent.graph_id)?.status;
    if (activeTab === "listening") return status === "listening";
    if (activeTab === "scheduled") return status === "scheduled";
    if (activeTab === "idle") return status === "idle";
    return false;
  });

  if (filtered.length === 0) {
    return <EmptyMessage tab={activeTab} />;
  }

  const status =
    activeTab === "listening"
      ? ("listening" as const)
      : activeTab === "scheduled"
        ? ("scheduled" as const)
        : ("idle" as const);

  const visible = showAll ? filtered : filtered.slice(0, MAX_VISIBLE);
  const hasMore = filtered.length > MAX_VISIBLE;

  return (
    <div>
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        {visible.map((agent) => (
          <SitrepItem
            key={agent.id}
            item={{
              id: agent.id,
              agentID: agent.id,
              agentName: agent.name,
              agentImageUrl: agent.image_url,
              priority: status,
              message: getAgentStatusLabel(activeTab, agent),
              status,
            }}
          />
        ))}
      </div>
      {hasMore && (
        <div className="mt-3 flex justify-center">
          <Button
            variant="secondary"
            size="small"
            onClick={() => setShowAll(!showAll)}
          >
            {showAll ? "Collapse" : `Show all (${filtered.length})`}
          </Button>
        </div>
      )}
    </div>
  );
}

function UsageFooter({
  usage,
  hasInsufficientCredits,
  onCreditChange,
}: {
  usage: CoPilotUsageStatus;
  hasInsufficientCredits: boolean;
  onCreditChange?: () => void;
}) {
  const isDailyExhausted =
    usage.daily.limit > 0 && usage.daily.used >= usage.daily.limit;
  const isWeeklyExhausted =
    usage.weekly.limit > 0 && usage.weekly.used >= usage.weekly.limit;
  const resetCost = usage.reset_cost ?? 0;
  const { resetUsage, isPending } = useResetRateLimit({ onCreditChange });

  const showReset =
    isDailyExhausted &&
    !isWeeklyExhausted &&
    resetCost > 0 &&
    !hasInsufficientCredits;

  const showAddCredits =
    isDailyExhausted && !isWeeklyExhausted && hasInsufficientCredits;

  if (!showReset && !showAddCredits) return null;

  return (
    <div className="mt-4 flex items-center gap-3">
      {showReset && (
        <Button
          variant="primary"
          size="small"
          onClick={() => resetUsage()}
          loading={isPending}
        >
          {isPending
            ? "Resetting..."
            : `Reset daily limit for ${formatCents(resetCost)}`}
        </Button>
      )}
      {showAddCredits && (
        <Link
          href="/profile/credits"
          className="inline-flex items-center justify-center rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90"
        >
          Add credits to reset
        </Link>
      )}
    </div>
  );
}

function UsageMeter({
  label,
  used,
  limit,
  resetsAt,
}: {
  label: string;
  used: number;
  limit: number;
  resetsAt: Date | string;
}) {
  if (limit <= 0) return null;

  const rawPercent = (used / limit) * 100;
  const percent = Math.min(100, Math.round(rawPercent));
  const isHigh = percent >= 80;
  const percentLabel =
    used > 0 && percent === 0 ? "<1% used" : `${percent}% used`;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between">
        <Text variant="body-medium" className="text-neutral-700">
          {label}
        </Text>
        <Text variant="body" className="tabular-nums text-neutral-500">
          {percentLabel}
        </Text>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-neutral-200">
        <div
          className={`h-full rounded-full transition-[width] duration-300 ease-out ${
            isHigh ? "bg-orange-500" : "bg-blue-500"
          }`}
          style={{ width: `${Math.max(used > 0 ? 1 : 0, percent)}%` }}
        />
      </div>
      <div className="flex items-baseline justify-between">
        <Text variant="small" className="tabular-nums text-neutral-500">
          {used.toLocaleString()} / {limit.toLocaleString()}
        </Text>
        <Text variant="small" className="text-neutral-400">
          Resets {formatResetTime(resetsAt)}
        </Text>
      </div>
    </div>
  );
}

const EMPTY_MESSAGES: Record<string, string> = {
  running: "No agents running right now",
  attention: "No agents that need attention",
  completed: "No recently completed runs",
  listening: "No agents listening for events",
  scheduled: "No agents with scheduled runs",
  idle: "No idle agents",
};

function EmptyMessage({ tab }: { tab: AgentStatusFilter }) {
  return (
    <div className="flex items-center justify-center pt-4">
      <Text variant="body-medium" className="text-zinc-600">
        {EMPTY_MESSAGES[tab] ?? "No agents in this category"}
      </Text>
    </div>
  );
}
