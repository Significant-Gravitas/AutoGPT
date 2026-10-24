"use client";

import { SecurityCheckIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  DelegationList,
  type DelegationPages,
} from "../../../components/DelegationList/DelegationList";
import {
  getDelegationMeta,
  getModeLabel,
  getOttoSummaryLine,
  OTTO_DELEGATION_FILTERS,
} from "../../../components/DelegationList/helpers";

interface Props {
  pages: DelegationPages;
  todayCount: number;
  mode: string | null;
  onChangeMode: () => void;
}

export function AutopilotDelegationsSection({
  pages,
  todayCount,
  mode,
  onChangeMode,
}: Props) {
  const modeLabel = getModeLabel(mode);
  const summary = pages.summary;

  return (
    <section className="flex flex-col gap-4 pt-2">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <Text variant="body" tone="primary" data-testid="delegations-summary">
          {summary ? getOttoSummaryLine(todayCount, summary) : " "}
        </Text>
        <div className="flex items-center gap-1.5">
          <Icon
            icon={SecurityCheckIcon}
            size={14}
            className="text-zinc-900"
            aria-hidden="true"
          />
          {modeLabel ? (
            <Text variant="small" as="span" tone="muted">
              {modeLabel}
            </Text>
          ) : null}
          <button
            type="button"
            onClick={onChangeMode}
            className="rounded-sm font-sans text-xs leading-[1.125rem] text-zinc-600 underline underline-offset-2 hover:text-zinc-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400"
          >
            Change
          </button>
        </div>
      </div>

      <DelegationList
        label="Delegations"
        pages={pages}
        filters={OTTO_DELEGATION_FILTERS}
        getMeta={getDelegationMeta}
        emptyMessage="No hand-offs yet. When Otto hands work to an expert, it shows up here."
      />
    </section>
  );
}
