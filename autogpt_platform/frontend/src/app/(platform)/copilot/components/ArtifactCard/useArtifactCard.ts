import { toast } from "@/components/molecules/Toast/use-toast";
import { useEffect } from "react";
import type { ArtifactRef } from "../../store";
import { useCopilotUIStore } from "../../store";
import { downloadArtifact } from "../ArtifactPanel/downloadArtifact";
import { classifyArtifact } from "../ArtifactPanel/helpers";

export function useArtifactCard(artifact: ArtifactRef, readOnly?: boolean) {
  const isActive = useCopilotUIStore(
    (s) => s.artifactPanel.activeArtifact?.id === artifact.id,
  );
  const openArtifact = useCopilotUIStore((s) => s.openArtifact);
  const registerArtifactForAutoOpen = useCopilotUIStore(
    (s) => s.registerArtifactForAutoOpen,
  );

  // Register this artifact on mount — the store decides whether to auto-open.
  // Fires once per artifact ID; subsequent renders with the same ID are no-ops
  // in the store (knownIds check).
  // Skipped in readOnly mode — the share viewer has no panel for auto-open
  // to target, and registering would just pollute the store.
  useEffect(() => {
    if (readOnly) return;
    registerArtifactForAutoOpen(artifact);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- re-register on ID or MIME change
  }, [artifact.id, artifact.mimeType, registerArtifactForAutoOpen, readOnly]);

  const classification = classifyArtifact(
    artifact.mimeType,
    artifact.title,
    artifact.sizeBytes,
  );

  function handleDownloadOnly() {
    downloadArtifact(artifact).catch(() => {
      toast({
        title: "Download failed",
        description: "Couldn't fetch the file.",
        variant: "destructive",
      });
    });
  }

  function handleOpen() {
    if (classification.openable) {
      openArtifact(artifact);
    } else {
      handleDownloadOnly();
    }
  }

  return {
    isActive,
    classification,
    handleOpen,
    handleDownloadOnly,
  };
}
