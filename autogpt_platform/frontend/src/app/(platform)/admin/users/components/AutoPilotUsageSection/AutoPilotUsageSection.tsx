"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { useState } from "react";
import { AdminSection } from "../../../components/AdminSection";
import { UsageBar } from "../../../components/UsageBar";
import { getPlanName } from "../../helpers";
import { ChangePlanDialog } from "./components/ChangePlanDialog";
import { PlanSelect } from "./components/PlanSelect";
import { ResetControls } from "./components/ResetControls";
import { useAutoPilotUsage } from "./useAutoPilotUsage";

interface Props {
  userId: string;
}

export function AutoPilotUsageSection({ userId }: Props) {
  const usage = useAutoPilotUsage(userId);
  const [pendingTier, setPendingTier] = useState<string | null>(null);
  const data = usage.rateLimit;

  return (
    <AdminSection
      title="AutoPilot usage"
      description="Daily and weekly spend against this user's plan limits."
    >
      {usage.isLoading ? (
        <div className="flex flex-col gap-4" aria-busy="true">
          <Skeleton className="h-9 w-[260px] rounded-lg" />
          <Skeleton className="h-12 w-full rounded-lg" />
          <Skeleton className="h-12 w-full rounded-lg" />
        </div>
      ) : !data ? (
        <div className="flex items-center justify-between gap-4">
          <Text variant="small" className="text-foreground-tertiary">
            Couldn&apos;t load this user&apos;s usage.
          </Text>
          <Button
            variant="secondary"
            size="small"
            onClick={() => void usage.refetch()}
          >
            Retry
          </Button>
        </div>
      ) : (
        <div className="flex flex-col gap-5">
          <PlanSelect
            tier={data.tier}
            multipliers={data.tier_multipliers}
            disabled={usage.isChangingPlan}
            onPick={setPendingTier}
          />
          <UsageRow
            label="Daily"
            used={data.daily_cost_used_microdollars}
            limit={data.daily_cost_limit_microdollars}
          />
          <UsageRow
            label="Weekly"
            used={data.weekly_cost_used_microdollars}
            limit={data.weekly_cost_limit_microdollars}
          />
          <ResetControls
            dailyUsed={data.daily_cost_used_microdollars}
            weeklyUsed={data.weekly_cost_used_microdollars}
            isResetting={usage.isResetting}
            onReset={usage.resetUsage}
          />
        </div>
      )}

      <ChangePlanDialog
        planName={pendingTier ? getPlanName(pendingTier) : ""}
        isOpen={pendingTier !== null}
        isChanging={usage.isChangingPlan}
        onCancel={() => setPendingTier(null)}
        onConfirm={() => {
          if (pendingTier) {
            usage.changePlan(pendingTier, () => setPendingTier(null));
          }
        }}
      />
    </AdminSection>
  );
}

function UsageRow({
  label,
  used,
  limit,
}: {
  label: string;
  used: number;
  limit: number;
}) {
  return (
    <div className="flex flex-col gap-1">
      <Text variant="small-medium" as="span">
        {label}
      </Text>
      <UsageBar used={used} limit={limit} />
    </div>
  );
}
