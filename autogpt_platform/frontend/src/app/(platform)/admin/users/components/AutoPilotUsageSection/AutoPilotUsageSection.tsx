"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Select } from "@/components/atoms/Select/Select";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { useState } from "react";
import { AdminSection } from "../../../components/AdminSection";
import { UsageBar } from "../../../components/UsageBar";
import { getPlanName } from "../../helpers";
import { ChangePlanDialog } from "./components/ChangePlanDialog";
import { ResetUsageDialog } from "./components/ResetUsageDialog";
import {
  getPlanOptions,
  isAssignableTier,
  RESET_SCOPE_OPTIONS,
  type ResetScope,
} from "./helpers";
import { useAutoPilotUsage } from "./useAutoPilotUsage";

interface Props {
  userId: string;
}

export function AutoPilotUsageSection({ userId }: Props) {
  const usage = useAutoPilotUsage(userId);
  const [pendingTier, setPendingTier] = useState<string | null>(null);
  const [resetScope, setResetScope] = useState<ResetScope>("daily");
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
          <Text variant="small" className="text-zinc-500">
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
          <div className="flex flex-wrap items-center justify-between gap-2">
            <Text variant="small-medium" as="span" aria-hidden>
              Plan
            </Text>
            <Select
              id="user-plan"
              label="Plan"
              hideLabel
              size="small"
              value={isAssignableTier(data.tier) ? data.tier : undefined}
              placeholder={`${getPlanName(data.tier)} (set by billing)`}
              options={getPlanOptions(data.tier_multipliers)}
              onValueChange={(tier) => {
                if (tier !== data.tier) setPendingTier(tier);
              }}
              disabled={usage.isChangingPlan}
              wrapperClassName="mb-0 w-full sm:w-[260px]"
            />
          </div>

          {[
            {
              label: "Daily",
              used: data.daily_cost_used_microdollars,
              limit: data.daily_cost_limit_microdollars,
            },
            {
              label: "Weekly",
              used: data.weekly_cost_used_microdollars,
              limit: data.weekly_cost_limit_microdollars,
            },
          ].map((bar) => (
            <div key={bar.label} className="flex flex-col gap-1">
              <Text variant="small-medium" as="span">
                {bar.label}
              </Text>
              <UsageBar used={bar.used} limit={bar.limit} />
            </div>
          ))}

          <div className="flex flex-wrap items-center justify-end gap-2">
            <Select
              id="reset-scope"
              label="What to reset"
              hideLabel
              size="small"
              value={resetScope}
              onValueChange={(value) => setResetScope(value as ResetScope)}
              options={RESET_SCOPE_OPTIONS}
              wrapperClassName="mb-0 w-[220px]"
            />
            <ResetUsageDialog
              scope={resetScope}
              nothingToReset={
                data.daily_cost_used_microdollars === 0 &&
                (resetScope === "daily" ||
                  data.weekly_cost_used_microdollars === 0)
              }
              isResetting={usage.isResetting}
              onConfirm={usage.resetUsage}
            />
          </div>
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
