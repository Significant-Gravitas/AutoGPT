"use client";

import { useContext } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { ExpertConfirmationContext } from "../ExpertConfirmationContext";
import type { TeamProposal } from "../helpers";
import { ProposedExpertRow } from "./components/ProposedExpertRow";
import { useTeamPreviewCard } from "./useTeamPreviewCard";

interface Props {
  proposals: TeamProposal[];
  founderMode?: boolean;
}

export function TeamPreviewCard({ proposals, founderMode = false }: Props) {
  const appliedConfirmationIDs = useContext(ExpertConfirmationContext);
  const {
    canConfirm,
    confirmed,
    kept,
    openId,
    removedIds,
    hireSelected,
    toggleOpen,
    toggleRemoved,
  } = useTeamPreviewCard(proposals, appliedConfirmationIDs);
  const allConfirmed = confirmed.length === proposals.length;

  return (
    <div className="rounded-2xl bg-white p-4 ring-1 ring-zinc-200/70">
      <div className="flex items-baseline justify-between gap-3">
        <p className="text-sm font-medium text-zinc-800">
          {proposals.length} experts for your team
        </p>
        <p className="shrink-0 text-xs text-zinc-400">
          {allConfirmed
            ? "Team ready"
            : confirmed.length > 0
              ? `${confirmed.length} Hired · ${kept.length} selected`
              : kept.length === proposals.length
                ? "Nothing created yet"
                : `${kept.length} of ${proposals.length} selected`}
        </p>
      </div>
      <div className="mt-1.5 flex flex-col divide-y divide-zinc-100">
        {proposals.map((proposal) => (
          <ProposedExpertRow
            key={proposal.confirmationId}
            proposal={proposal}
            confirmed={appliedConfirmationIDs.has(proposal.confirmationId)}
            removed={removedIds.has(proposal.confirmationId)}
            open={openId === proposal.confirmationId}
            founderMode={founderMode}
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
            onClick={hireSelected}
          >
            Hire selected
          </Button>
          <span className="text-xs text-zinc-400">
            {kept.length === 0
              ? "Put someone back, or tell me who to draft instead."
              : "To change a charter, say what should be different."}
          </span>
        </div>
      )}
    </div>
  );
}
