"use client";

import { LiveDelegationProbes } from "../../DelegationStatusLine/LiveDelegationProbes";
import {
  useChatSessionDelegations,
  useLiveStatuses,
} from "../useChatSessionDelegations";

interface Props {
  sessionId: string | null;
}

/** How many hand-offs wait on the user: approvals plus questions. */
export function WorkBadge({ sessionId }: Props) {
  const { delegations } = useChatSessionDelegations(sessionId);
  const { liveStatuses, reportStatus } = useLiveStatuses();
  const waiting = delegations.filter((delegation) => {
    const status = liveStatuses[delegation.toolCallId] ?? delegation.status;
    return status === "proposed" || status === "needs-input";
  }).length;
  return (
    <>
      <LiveDelegationProbes delegations={delegations} onStatus={reportStatus} />
      {waiting > 0 && (
        <span
          data-testid="work-badge"
          aria-label={`${waiting} waiting on you`}
          className="inline-flex min-w-4 items-center justify-center rounded-full bg-amber-100 px-1.5 text-[11px] font-medium leading-4 text-amber-800"
        >
          {waiting}
        </span>
      )}
    </>
  );
}
