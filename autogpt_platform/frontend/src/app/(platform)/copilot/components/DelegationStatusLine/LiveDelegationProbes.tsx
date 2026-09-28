"use client";

import { useEffect } from "react";
import type { ChatDelegation, LiveDelegationStatus } from "../../delegations";
import { useDelegationLive } from "../../useDelegationLive";

interface ProbeProps {
  delegation: ChatDelegation;
  onStatus: (toolCallId: string, status: LiveDelegationStatus) => void;
}

function LiveDelegationProbe({ delegation, onStatus }: ProbeProps) {
  const { status } = useDelegationLive(delegation);
  useEffect(() => {
    onStatus(delegation.toolCallId, status);
  }, [delegation.toolCallId, status, onStatus]);
  return null;
}

interface Props {
  delegations: ChatDelegation[];
  onStatus: (toolCallId: string, status: LiveDelegationStatus) => void;
}

/** One invisible poller per hand-off. The list grows while a turn streams,
 *  so the polls cannot be hooks in a loop; each one is its own component and
 *  reports its live status up to whoever needs the counts. */
export function LiveDelegationProbes({ delegations, onStatus }: Props) {
  return delegations.map((delegation) => (
    <LiveDelegationProbe
      key={delegation.toolCallId}
      delegation={delegation}
      onStatus={onStatus}
    />
  ));
}
