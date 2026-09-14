"use client";

import type { RaiseAttachmentDraft } from "../../helpers";
import { AttachmentStep } from "../KitStep/AttachmentStep";

interface Props {
  color: string | null;
  submitted: RaiseAttachmentDraft[] | null;
  onSubmit: (attachments: RaiseAttachmentDraft[]) => void;
  onSkip: () => void;
}

export function MarketplaceStep({ color, submitted, onSubmit, onSkip }: Props) {
  return (
    <AttachmentStep
      color={color}
      submitted={submitted}
      scope="marketplace"
      searchLabel="Search marketplace and library workflows"
      searchPlaceholder="Search marketplace and library workflows…"
      emptyQueryHint="Search to add marketplace or library workflows."
      emptyResultsHint="No matching workflows."
      primaryLabel="That's it"
      onSubmit={onSubmit}
      onSkip={onSkip}
    />
  );
}
