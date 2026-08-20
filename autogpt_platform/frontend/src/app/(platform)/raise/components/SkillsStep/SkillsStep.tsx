"use client";

import type { RaiseAttachmentDraft } from "../../helpers";
import { AttachmentStep } from "../KitStep/AttachmentStep";

interface Props {
  name: string;
  color: string | null;
  submitted: RaiseAttachmentDraft[] | null;
  existingCount: number;
  isSubmitting: boolean;
  onSubmit: (attachments: RaiseAttachmentDraft[]) => void;
  onSkip: () => void;
}

export function SkillsStep({
  name,
  color,
  submitted,
  existingCount,
  isSubmitting,
  onSubmit,
  onSkip,
}: Props) {
  return (
    <AttachmentStep
      color={color}
      submitted={submitted}
      isSubmitting={isSubmitting}
      scope="skills"
      existingCount={existingCount}
      searchLabel="Search skills"
      searchPlaceholder="Search library and marketplace skills…"
      emptyQueryHint="Search to add a marketplace skill, or pick from your library skills."
      emptyResultsHint="No matching skills."
      primaryLabel={`Bring ${name || "it"} to life`}
      onSubmit={onSubmit}
      onSkip={onSkip}
    />
  );
}
