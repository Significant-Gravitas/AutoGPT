"use client";

import { useContext, useState } from "react";
import { CopilotChatActionsContext } from "../../CopilotChatActionsProvider/useCopilotChatActions";
import type { TeamProposal } from "../helpers";
import { buildConfirmMessage } from "./helpers";

export function useTeamPreviewCard(
  proposals: TeamProposal[],
  appliedConfirmationIDs: ReadonlySet<string>,
) {
  const actions = useContext(CopilotChatActionsContext);
  const [removedIds, setRemovedIds] = useState<readonly string[]>([]);
  const [openId, setOpenId] = useState<string | null>(null);
  const confirmed = proposals.filter((proposal) =>
    appliedConfirmationIDs.has(proposal.confirmationId),
  );
  const pending = proposals.filter(
    (proposal) => !appliedConfirmationIDs.has(proposal.confirmationId),
  );
  const kept = pending.filter(
    (proposal) => !removedIds.includes(proposal.confirmationId),
  );

  function toggleRemoved(confirmationId: string) {
    setRemovedIds((previous) =>
      previous.includes(confirmationId)
        ? previous.filter((id) => id !== confirmationId)
        : [...previous, confirmationId],
    );
  }

  function toggleOpen(confirmationId: string) {
    setOpenId((previous) =>
      previous === confirmationId ? null : confirmationId,
    );
  }

  function hireSelected() {
    if (!actions || kept.length === 0) return;
    actions.onSend(buildConfirmMessage(kept));
  }

  return {
    canConfirm: actions !== null && pending.length > 0,
    confirmed,
    kept,
    openId,
    removedIds,
    hireSelected,
    toggleOpen,
    toggleRemoved,
  };
}
