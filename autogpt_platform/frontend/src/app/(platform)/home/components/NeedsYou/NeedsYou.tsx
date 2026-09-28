"use client";

import { AlertCircleIcon } from "@hugeicons/core-free-icons";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import { Text } from "@/components/atoms/Text/Text";
import { AUTOPILOT_NAME } from "@/components/molecules/AutopilotAvatar/helpers";
import { HomeTileFilter } from "../HomeTileFilter/HomeTileFilter";
import { HomeTile } from "../HomeTile/HomeTile";
import { RejectAllFooter } from "@/app/(platform)/copilot/components/ApprovalQueue/components/RejectAllFooter";
import { AttentionRow } from "./components/AttentionRow";
import { GroupHeader } from "./components/GroupHeader";
import { HeldCall } from "./components/HeldCall";
import {
  type AttentionGroup,
  canApproveGroup,
  isHeldCall,
  undecidedHeldCalls,
} from "./helpers";
import type { AttentionListRow } from "./useHeldReview";
import { useNeedsYou } from "./useNeedsYou";

interface Props {
  dashboard: HomeDashboardResponse;
  className?: string;
}

export function NeedsYou({ dashboard, className }: Props) {
  const {
    groups,
    rejectable,
    confirmRejectAll,
    setConfirmRejectAll,
    pendingCount,
    filterOptions,
    hasFilters,
    selectedKind,
    selectKind,
    pendingIDs,
    decide,
    held,
  } = useNeedsYou({ items: dashboard.attention });

  return (
    <HomeTile
      className={className}
      icon={AlertCircleIcon}
      title="Needs you"
      badge={
        <Text
          variant="small-medium"
          as="span"
          tone="secondary"
          role="status"
          aria-label={`${pendingCount} ${pendingCount === 1 ? "item needs" : "items need"} your attention`}
          className="rounded-md bg-zinc-100 px-1.5 py-0.5 tabular-nums"
        >
          {pendingCount}
        </Text>
      }
      meta={
        hasFilters ? (
          <HomeTileFilter
            ariaLabelPrefix="Filter interventions"
            value={selectedKind}
            options={filterOptions}
            onChange={(value) => selectKind(value as typeof selectedKind)}
          />
        ) : null
      }
    >
      <div className="divide-y divide-zinc-100">
        {groups.map((group) =>
          group.rows.length < 2 ? (
            renderRow(group.rows[0], 40)
          ) : (
            <section
              key={group.key}
              aria-label={group.expert?.name ?? AUTOPILOT_NAME}
              className="divide-y divide-zinc-100"
            >
              {renderHeader(group)}
              {group.rows.map((row) => renderRow(row, null))}
            </section>
          ),
        )}
      </div>
      {rejectable.length >= 2 && (
        <RejectAllFooter
          count={rejectable.length}
          confirming={confirmRejectAll}
          busy={anyBusy(rejectable.map((item) => item.id))}
          told="Each Expert will be told"
          onAsk={() => setConfirmRejectAll(true)}
          onCancel={() => setConfirmRejectAll(false)}
          onConfirm={() => {
            setConfirmRejectAll(false);
            held.decide(rejectable, false);
          }}
        />
      )}
      <span className="sr-only" aria-live="polite">
        {held.announcement}
      </span>
    </HomeTile>
  );

  function renderRow(
    { item, receipt }: AttentionListRow,
    avatar: number | null,
  ) {
    return isHeldCall(item) ? (
      <HeldCall
        key={item.id}
        item={item}
        receipt={receipt}
        status={held.statusOf(item.id)}
        failed={held.hasFailed(item.id)}
        open={held.openId === item.id}
        avatarSize={avatar}
        onToggle={() => held.toggle(item.id)}
        onClose={() => held.close(item.id)}
        onDecide={(approved, rule, scope) =>
          held.decide([item], approved, rule, scope)
        }
      />
    ) : (
      <AttentionRow
        key={item.id}
        item={item}
        isProcessing={pendingIDs.has(item.id)}
        onDecision={decide}
      />
    );
  }

  function renderHeader(group: AttentionGroup) {
    const calls = undecidedHeldCalls(group.rows);
    return (
      <GroupHeader
        key={`${group.key}-header`}
        group={group}
        pending={group.rows.filter((row) => !row.receipt).length}
        rejectable={calls.length}
        approvable={canApproveGroup(calls) ? calls.length : 0}
        busy={anyBusy(calls.map((item) => item.id))}
        onRejectAll={() => held.decide(calls, false)}
        onApproveAll={() => held.decide(calls, true)}
      />
    );
  }

  function anyBusy(ids: string[]) {
    return ids.some((id) => held.statusOf(id) !== "idle");
  }
}
