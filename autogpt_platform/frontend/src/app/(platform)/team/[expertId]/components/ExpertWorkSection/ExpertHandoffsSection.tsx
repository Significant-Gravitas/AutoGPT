"use client";

import { Text } from "@/components/atoms/Text/Text";
import { DelegationList } from "../../../components/DelegationList/DelegationList";
import {
  EXPERT_DELEGATION_FILTERS,
  getExpertSummaryLine,
  getHandoffMeta,
} from "../../../components/DelegationList/helpers";
import { useExpertHandoffs } from "./useExpertHandoffs";

interface Props {
  expertId: string;
  expertName: string;
  enabled: boolean;
}

export function ExpertHandoffsSection({
  expertId,
  expertName,
  enabled,
}: Props) {
  const { delegations, isLoading, isError, refetch } = useExpertHandoffs({
    expertId,
    enabled,
  });
  const allFromOtto =
    delegations.length > 0 &&
    delegations.every((d) => !d.delegated_by_expert_id);

  return (
    <section aria-label="Hand-offs from Otto" className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-4 pt-2">
        <Text variant="body" tone="primary" data-testid="handoffs-summary">
          {isLoading || isError
            ? "Hand-offs"
            : getExpertSummaryLine(delegations)}
        </Text>
        {allFromOtto ? (
          <Text variant="small" as="span" tone="muted">
            All from Otto
          </Text>
        ) : null}
      </div>
      <DelegationList
        label="Hand-offs"
        delegations={delegations}
        filters={EXPERT_DELEGATION_FILTERS}
        getMeta={(delegation) => getHandoffMeta(delegation)}
        emptyMessage={`Nothing handed to ${expertName} yet. Work Otto delegates shows up here.`}
        isLoading={isLoading}
        isError={isError}
        onRetry={() => void refetch()}
      />
    </section>
  );
}
