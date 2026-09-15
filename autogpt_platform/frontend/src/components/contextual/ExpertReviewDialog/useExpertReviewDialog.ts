import type { ExpertPackagePreview } from "@/app/api/__generated__/models/expertPackagePreview";
import { useEffect, useState } from "react";
import {
  buildDraftFromPreview,
  buildEditsFromDraft,
  getBlockingReason,
  toggleMember,
  type ExpertReviewDraft,
} from "./helpers";

interface Args {
  mode: "import" | "publish";
  open: boolean;
  preview: ExpertPackagePreview | null;
}

export function useExpertReviewDialog({ mode, open, preview }: Args) {
  const [draft, setDraft] = useState<ExpertReviewDraft>(() =>
    buildDraftFromPreview(preview),
  );

  // The preview arrives after the dialog opens (the file is parsed on the
  // server), and a second file must not inherit the first one's edits.
  useEffect(() => {
    if (open) setDraft(buildDraftFromPreview(preview));
  }, [open, preview]);

  function setName(name: string) {
    setDraft((current) => ({ ...current, name }));
  }

  function toggleSkill(slug: string) {
    setDraft((current) => ({
      ...current,
      removedSkillSlugs: toggleMember(current.removedSkillSlugs, slug),
    }));
  }

  function toggleWorkflow(index: number) {
    setDraft((current) => ({
      ...current,
      removedWorkflowIndices: toggleMember(
        current.removedWorkflowIndices,
        index,
      ),
    }));
  }

  function toggleSchedule(index: number) {
    setDraft((current) => ({
      ...current,
      scheduledIndices: toggleMember(current.scheduledIndices, index),
    }));
  }

  return {
    draft,
    setName,
    toggleSkill,
    toggleWorkflow,
    toggleSchedule,
    edits: buildEditsFromDraft(draft, preview),
    blockingReason: getBlockingReason(mode, draft, preview),
  };
}
