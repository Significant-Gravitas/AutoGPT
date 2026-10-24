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
}

/** On the Overview, once Otto has handed this expert anything: an expert
 *  nobody delegated to keeps the page it had. */
export function ExpertHandoffsSection({ expertId, expertName }: Props) {
  const pages = useExpertHandoffs(expertId);
  if (pages.isLoading || (!pages.isError && pages.total === 0)) return null;
  const allFromOtto =
    pages.delegations.length > 0 &&
    pages.delegations.every((d) => !d.delegated_by_expert_id);

  return (
    <section aria-label="Hand-offs from Otto" className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-4 pt-2">
        <Text variant="body" tone="primary" data-testid="handoffs-summary">
          {pages.summary ? getExpertSummaryLine(pages.summary) : "Hand-offs"}
        </Text>
        {allFromOtto ? (
          <Text variant="small" as="span" tone="muted">
            All from Otto
          </Text>
        ) : null}
      </div>
      <DelegationList
        label="Hand-offs"
        pages={pages}
        filters={EXPERT_DELEGATION_FILTERS}
        getMeta={(delegation) => getHandoffMeta(delegation)}
        emptyMessage={`Nothing handed to ${expertName} yet. Work Otto delegates shows up here.`}
      />
    </section>
  );
}
