"use client";

import type { ChatDelegation } from "../../../../../delegations";
import {
  delegationTitle,
  getDelegationStatusView,
  threadHref,
} from "../../../../../delegationViews";
import { useDelegationLive } from "../../../../../useDelegationLive";
import { HandoffMeta } from "../../../../HandoffApprovalCard/components/HandoffMeta";
import { delegationSubtitle } from "../helpers";
import { buildTimeline } from "../timeline";
import { useDelegationControls } from "../useDelegationControls";
import { DelegationTimeline } from "./DelegationTimeline";
import { DetailActions } from "./DetailActions";
import { DetailHeader } from "./DetailHeader";
import { DetailSection } from "./DetailSection";
import { SubSessionReviews } from "./SubSessionReviews";

interface Props {
  delegation: ChatDelegation;
  chatSessionId: string | null;
  onBack: () => void;
}

export function DelegationDetail({ delegation, chatSessionId, onBack }: Props) {
  const live = useDelegationLive(delegation);
  const name = live.expert.name;
  const controls = useDelegationControls(
    delegation.subSessionId,
    name,
    chatSessionId,
  );

  return (
    <div className="flex flex-col gap-4 p-4" data-testid="delegation-detail">
      <DetailHeader
        expert={live.expert}
        title={delegationTitle(delegation, name)}
        subtitle={delegationSubtitle(delegation, {
          status: live.status,
          elapsedSeconds: live.elapsedSeconds,
          askedAt: live.askedAt,
          resumed: !!live.answer,
        })}
        view={getDelegationStatusView(live.status)}
        onBack={onBack}
      />
      <DetailActions
        delegation={delegation}
        live={live}
        href={threadHref(delegation)}
        controls={controls}
      />
      <SubSessionReviews
        subSessionId={delegation.subSessionId}
        expertName={name}
      />
      {delegation.prompt && (
        <DetailSection title="Brief">
          <p className="line-clamp-8 whitespace-pre-wrap text-sm text-zinc-700">
            {delegation.prompt}
          </p>
        </DetailSection>
      )}
      <HandoffMeta
        facts={[
          { label: "Owner", value: "You" },
          { label: "Expected back", value: "A report in this chat" },
        ]}
      />
      <DetailSection title="Timeline">
        <DelegationTimeline
          entries={buildTimeline(live.session, delegation, live.status)}
        />
      </DetailSection>
    </div>
  );
}
