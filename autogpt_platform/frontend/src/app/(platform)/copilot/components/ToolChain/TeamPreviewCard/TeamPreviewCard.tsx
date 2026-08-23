"use client";

import { Button } from "@/components/atoms/Button/Button";
import type { TeamProposal } from "../helpers";
import { ProposedExpertRow } from "./components/ProposedExpertRow";
import { useTeamPreviewCard } from "./useTeamPreviewCard";

interface Props {
  proposals: TeamProposal[];
}

/** A whole team proposed in one turn, as one roster the user confirms once.
 *  Each teammate opens to their full charter and can be dropped before
 *  confirming; changing a charter stays conversational — the user says what
 *  to change and the model re-proposes. */
export function TeamPreviewCard({ proposals }: Props) {
  const {
    canConfirm,
    kept,
    openId,
    removedIds,
    hireAll,
    toggleOpen,
    toggleRemoved,
  } = useTeamPreviewCard(proposals);

  return (
    <div className="rounded-2xl bg-white p-4 ring-1 ring-zinc-200/70">
      <div className="flex items-baseline justify-between gap-3">
        <p className="text-sm font-medium text-zinc-800">
          {proposals.length} experts for your team
        </p>
        <p className="shrink-0 text-xs text-zinc-400">
          {kept.length === proposals.length
            ? "Nothing created yet"
            : `${kept.length} of ${proposals.length} selected`}
        </p>
      </div>
      <div className="mt-1.5 flex flex-col divide-y divide-zinc-100">
        {proposals.map((proposal) => (
          <ProposedExpertRow
            key={proposal.confirmationId}
            proposal={proposal}
            removed={removedIds.includes(proposal.confirmationId)}
            open={openId === proposal.confirmationId}
            onToggleRemoved={() => toggleRemoved(proposal.confirmationId)}
            onToggleOpen={() => toggleOpen(proposal.confirmationId)}
          />
        ))}
      </div>
      {canConfirm && (
        <div className="mt-3 flex flex-wrap items-center gap-2.5">
          <Button
            variant="primary"
            size="small"
            disabled={kept.length === 0}
            onClick={hireAll}
          >
            Hire all
          </Button>
          <span className="text-xs text-zinc-400">
            {kept.length === 0
              ? "Put someone back, or tell me who to draft instead."
              : "To change a charter, just say what should be different."}
          </span>
        </div>
      )}
    </div>
  );
}
