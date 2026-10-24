"use client";

import { useState } from "react";
import type { HandoffEdits } from "./helpers";

/** The card's own edits before Approve: a rewritten brief, another expert. */
export function useHandoffApprovalCard(brief: string | null) {
  const [isEditing, setIsEditing] = useState(false);
  const [draft, setDraft] = useState(brief ?? "");
  const [pickedExpertId, setPickedExpertId] = useState<string | null>(null);
  const [isExpanded, setIsExpanded] = useState(false);

  const edits: HandoffEdits = {
    ...(isEditing && draft.trim() && draft !== brief
      ? { prompt: draft.trim() }
      : {}),
    ...(pickedExpertId ? { expert_id: pickedExpertId } : {}),
  };

  return {
    isEditing,
    toggleEditing: () => setIsEditing((value) => !value),
    draft,
    setDraft,
    pickedExpertId,
    pickExpert: setPickedExpertId,
    isExpanded,
    toggleExpanded: () => setIsExpanded((value) => !value),
    edits,
  };
}
