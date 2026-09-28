"use client";

import { UserGroupIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { AUTOPILOT_NAME } from "@/components/molecules/AutopilotAvatar/helpers";
import { LiveDelegationProbes } from "../../../DelegationStatusLine/LiveDelegationProbes";
import { DelegationDetail } from "./components/DelegationDetail";
import { DelegationRow } from "./components/DelegationRow";
import { WorkTotals } from "./components/WorkTotals";
import { countExperts, workTotals } from "./helpers";
import { useWorkTab } from "./useWorkTab";

interface Props {
  sessionId: string | null;
}

function EmptyWork() {
  return (
    <div className="flex flex-col items-center gap-2 px-6 py-12 text-center">
      <Icon icon={UserGroupIcon} size={24} className="text-zinc-400" />
      <Text variant="h5">Nothing delegated yet</Text>
      <Text variant="body" tone="secondary">
        When {AUTOPILOT_NAME} hands work to an expert, it shows up here with its
        status and what came back.
      </Text>
    </div>
  );
}

export function WorkTab({ sessionId }: Props) {
  const {
    delegations,
    liveStatuses,
    reportStatus,
    selected,
    select,
    isLoading,
    isError,
  } = useWorkTab(sessionId);

  if (isLoading) {
    return (
      <div className="flex flex-col gap-2 p-4">
        <Skeleton className="h-16 w-full rounded-xl" />
        <Skeleton className="h-16 w-full rounded-xl" />
      </div>
    );
  }
  if (isError) {
    return (
      <div className="p-3">
        <ErrorCard
          isSuccess={false}
          context="delegations"
          responseError={{ message: "Failed to load this chat's work." }}
        />
      </div>
    );
  }
  if (selected) {
    return (
      <div className="min-h-0 flex-1 overflow-y-auto">
        <DelegationDetail
          delegation={selected}
          chatSessionId={sessionId}
          onBack={() => select(null)}
        />
      </div>
    );
  }
  if (delegations.length === 0) return <EmptyWork />;

  const totals = workTotals(
    delegations,
    (d) => liveStatuses[d.toolCallId] ?? d.status,
  );
  const experts = countExperts(delegations);

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto p-4">
      <LiveDelegationProbes delegations={delegations} onStatus={reportStatus} />
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <Text variant="eyebrow">This chat</Text>
          <span className="text-xs text-zinc-500">
            {AUTOPILOT_NAME} → {experts} expert{experts === 1 ? "" : "s"}
          </span>
        </div>
        <div className="flex flex-col gap-1.5">
          {delegations.map((delegation) => (
            <DelegationRow
              key={delegation.toolCallId}
              delegation={delegation}
              onOpen={() => select(delegation.toolCallId)}
            />
          ))}
        </div>
      </div>
      <WorkTotals totals={totals} />
    </div>
  );
}
