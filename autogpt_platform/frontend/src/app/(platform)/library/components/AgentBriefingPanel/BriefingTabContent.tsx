"use client";

import type { CoPilotUsagePublic } from "@/app/api/__generated__/models/coPilotUsagePublic";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useGetV2GetCopilotUsage } from "@/app/api/__generated__/endpoints/chat/chat";
import { formatResetTime } from "@/app/(platform)/copilot/components/usageHelpers";
import { Button } from "@/components/atoms/Button/Button";
import { Badge } from "@/components/atoms/Badge/Badge";
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
  const { data: usage, isSuccess } = useGetV2GetCopilotUsage({
    query: {
      select: (res) => res.data as CoPilotUsagePublic,
      refetchInterval: 30000,
      staleTime: 10000,
    },
  });

  const isBillingEnabled = useGetFlag(Flag.ENABLE_PLATFORM_PAYMENT);

  if (!isSuccess || !usage) return null;
  if (!usage.daily && !usage.weekly) return null;

  const isDailyExhausted = !!usage.daily && usage.daily.percent_used >= 100;
  const showGoToBilling = isDailyExhausted && isBillingEnabled;

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
        {isBillingEnabled && !showGoToBilling && (
          <Link
            href="/settings/billing"
            className="text-sm text-blue-600 hover:underline"
          >
            Manage billing
          </Link>
        )}
      </div>
      <div className="mt-4 grid grid-cols-1 gap-6 sm:grid-cols-2">
        {usage.daily && (
          <UsageMeter
            label="Today"
            percentUsed={usage.daily.percent_used}
            resetsAt={usage.daily.resets_at}
          />
        )}
        {usage.weekly && (
          <UsageMeter
            label="This week"
            percentUsed={usage.weekly.percent_used}
            resetsAt={usage.weekly.resets_at}
          />
        )}
      </div>
      {showGoToBilling && (
        <div className="mt-4 flex items-center gap-3">
          <Button
            as="NextLink"
            href="/settings/billing"
            variant="primary"
            size="small"
          >
            Go to billing
          </Button>
        </div>
      )}
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

function UsageMeter({
  label,
  percentUsed,
  resetsAt,
}: {
  label: string;
  percentUsed: number;
  resetsAt: Date | string;
}) {
  const percent = Math.min(100, Math.max(0, Math.round(percentUsed)));
  const isHigh = percent >= 80;
  const percentLabel =
    percentUsed > 0 && percent === 0 ? "<1% used" : `${percent}% used`;

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
          role="progressbar"
          aria-label={`${label} usage`}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={percent}
          className={`h-full rounded-full transition-[width] duration-300 ease-out ${
            isHigh ? "bg-orange-500" : "bg-blue-500"
          }`}
          style={{ width: `${Math.max(percent > 0 ? 1 : 0, percent)}%` }}
        />
      </div>
      <Text variant="small" className="text-neutral-400">
        Resets {formatResetTime(resetsAt)}
      </Text>
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
