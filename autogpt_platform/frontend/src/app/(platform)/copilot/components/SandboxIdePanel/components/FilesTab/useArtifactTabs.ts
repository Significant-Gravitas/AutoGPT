"use client";

import { useEffect, useState } from "react";
import type { ArtifactRef } from "../../../../store";
import { useCopilotUIStore } from "../../../../store";

/**
 * Tracks the set of artifact previews open as tabs in the sandbox IDE. The
 * store only holds one `activeArtifact`; this keeps a local list (mirroring the
 * sandbox file tabs) so several artifacts can stay open at once and be switched
 * between, while the active one still drives the editor pane.
 */
export function useArtifactTabs() {
  const activeArtifact = useCopilotUIStore(
    (s) => s.artifactPanel.activeArtifact,
  );
  const openArtifact = useCopilotUIStore((s) => s.openArtifact);
  const clearArtifactPreview = useCopilotUIStore((s) => s.clearArtifactPreview);

  const [openArtifacts, setOpenArtifacts] = useState<ArtifactRef[]>([]);

  // Whenever an artifact becomes active (tree click, chat card, …) make sure it
  // has a tab.
  useEffect(() => {
    if (!activeArtifact) return;
    setOpenArtifacts((prev) =>
      prev.some((a) => a.id === activeArtifact.id)
        ? prev
        : [...prev, activeArtifact],
    );
  }, [activeArtifact]);

  function selectArtifact(ref: ArtifactRef) {
    openArtifact(ref);
  }

  function closeArtifact(ref: ArtifactRef) {
    const next = openArtifacts.filter((a) => a.id !== ref.id);
    setOpenArtifacts(next);
    // Closing the active tab falls back to the last remaining artifact, or
    // clears the preview entirely when none are left.
    if (activeArtifact?.id === ref.id) {
      if (next.length > 0) openArtifact(next[next.length - 1]);
      else clearArtifactPreview();
    }
  }

  return {
    openArtifacts,
    activeArtifactId: activeArtifact?.id ?? null,
    selectArtifact,
    closeArtifact,
  };
}
