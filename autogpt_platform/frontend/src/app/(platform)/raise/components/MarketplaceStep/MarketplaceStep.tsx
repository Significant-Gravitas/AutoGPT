"use client";

import type { RaiseAttachmentDraft } from "../../helpers";
import { AttachmentStep } from "../KitStep/AttachmentStep";

interface Props {
  name: string;
  color: string | null;
  submitted: RaiseAttachmentDraft[] | null;
  isFinal: boolean;
  isSubmitting: boolean;
  onSubmit: (attachments: RaiseAttachmentDraft[]) => void;
  onSkip: () => void;
}

export function MarketplaceStep({
  name,
  color,
  submitted,
  isFinal,
  isSubmitting,
  onSubmit,
  onSkip,
}: Props) {
  return (
    <AttachmentStep
      color={color}
      submitted={submitted}
      isSubmitting={isSubmitting}
      scope="marketplace"
      searchLabel="Search marketplace and library workflows"
      searchPlaceholder="Search marketplace and library workflows…"
      emptyQueryHint="Search to add marketplace or library workflows."
      emptyResultsHint="No matching workflows."
      primaryLabel={isFinal ? `Bring ${name || "them"} to life` : "That's it"}
      onSubmit={onSubmit}
      onSkip={onSkip}
    />
  );
}
