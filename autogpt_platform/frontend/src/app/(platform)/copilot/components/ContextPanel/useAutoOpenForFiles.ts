"use client";

import { useEffect, useRef } from "react";
import { useCopilotUIStore } from "../../store";
import { fileItemToArtifactRef } from "./components/FilesTab/helpers";
import {
  useSessionFiles,
  type SessionFile,
} from "./components/FilesTab/useSessionFiles";

function getLastGeneratedFile(generated: SessionFile[]): SessionFile | null {
  if (generated.length === 0) return null;
  return generated.reduce((latest, file) =>
    new Date(file.item.created_at).getTime() >
    new Date(latest.item.created_at).getTime()
      ? file
      : latest,
  );
}

/**
 * The first time a session is found to have generated files, opens the Artifact
 * panel directly on the most recently generated file (unless the user has
 * explicitly closed the panel). Mirrors useAutoOpenForProgress.
 */
export function useAutoOpenForFiles(sessionId: string | null) {
  const autoOpenArtifact = useCopilotUIStore((s) => s.autoOpenArtifact);
  const { generated } = useSessionFiles(sessionId);
  const lastGenerated = getLastGeneratedFile(generated);
  const triggered = useRef(false);
  const prevSessionIdRef = useRef(sessionId);

  // Session change: clear the per-session "already triggered" guard so the
  // next session can auto-open independently. The shared `_autoOpenReady` /
  // `_autoOpenUserClosed` flags are reset by `useAutoOpenArtifacts` (the
  // single owner of that global state); calling `resetAutoOpenState()` here
  // too races with that hook and can clobber the ready flag depending on
  // mount order.
  useEffect(() => {
    if (prevSessionIdRef.current !== sessionId) {
      triggered.current = false;
      prevSessionIdRef.current = sessionId;
    }
  }, [sessionId]);

  useEffect(() => {
    if (lastGenerated && !triggered.current) {
      triggered.current = true;
      autoOpenArtifact(fileItemToArtifactRef(lastGenerated.item));
    }
  }, [lastGenerated, autoOpenArtifact]);
}
