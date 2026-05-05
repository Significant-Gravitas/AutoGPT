"use client";

import { UIDataTypes, UIMessage, UITools } from "ai";
import { useEffect, useRef } from "react";
import { getMessageArtifacts } from "../ChatMessagesContainer/helpers";
import { useCopilotUIStore } from "../../store";

interface UseAutoOpenArtifactsOptions {
  sessionId: string | null;
  messages: UIMessage<unknown, UIDataTypes, UITools>[];
  isLoadingSession: boolean;
  isArtifactsEnabled: boolean;
}

export function useAutoOpenArtifacts({
  sessionId,
  messages,
  isLoadingSession,
  isArtifactsEnabled,
}: UseAutoOpenArtifactsOptions) {
  const resetArtifactPanel = useCopilotUIStore(
    (state) => state.resetArtifactPanel,
  );
  const openArtifact = useCopilotUIStore((state) => state.openArtifact);
  const isOpen = useCopilotUIStore((s) => s.artifactPanel.isOpen);

  const prevSessionIdRef = useRef(sessionId);
  const knownArtifactIdsRef = useRef<Set<string>>(new Set());
  const snapshotTakenRef = useRef(false);
  const userHasClosedRef = useRef(false);
  const wasOpenRef = useRef(false);

  // Detect the user explicitly closing the panel (isOpen: true → false within
  // the same session mount). Suppress auto-open for the remainder of this
  // session once detected. Component remounts on session change (key={sessionId}),
  // so these refs reset naturally — other sessions are unaffected.
  useEffect(() => {
    if (wasOpenRef.current && !isOpen) {
      userHasClosedRef.current = true;
    }
    wasOpenRef.current = isOpen;
  }, [isOpen]);

  // Session change: fully clear the panel so stale artifacts and back-stack
  // entries never bleed into the next chat. Kept defensively even though
  // ChatContainer uses key={sessionId} (which causes a full remount).
  useEffect(() => {
    const isSessionChange = prevSessionIdRef.current !== sessionId;
    prevSessionIdRef.current = sessionId;

    if (isSessionChange) {
      resetArtifactPanel();
    }
  }, [sessionId, resetArtifactPanel]);

  // Auto-open: watch messages for newly-created agent artifacts.
  useEffect(() => {
    if (!sessionId || isLoadingSession || !isArtifactsEnabled) return;

    if (userHasClosedRef.current) return;

    const agentArtifacts = messages
      .flatMap((msg) => getMessageArtifacts(msg))
      .filter((a) => a.origin === "agent");

    if (!snapshotTakenRef.current) {
      // First stable render for this session: snapshot existing artifacts so
      // we don't auto-open things the agent produced in a prior run.
      snapshotTakenRef.current = true;
      for (const a of agentArtifacts) {
        knownArtifactIdsRef.current.add(a.id);
      }
      return;
    }

    const newArtifacts = agentArtifacts.filter(
      (a) => !knownArtifactIdsRef.current.has(a.id),
    );

    for (const a of newArtifacts) {
      knownArtifactIdsRef.current.add(a.id);
    }

    if (newArtifacts.length === 0) return;

    // Open the most recently added artifact.
    openArtifact(newArtifacts[newArtifacts.length - 1]);
  }, [sessionId, isLoadingSession, isArtifactsEnabled, messages, openArtifact]);

  // Reset on unmount so navigating away from /copilot (to /profile, /home,
  // etc.) can't leave the panel open in the Zustand store, which would then
  // render the panel re-open when the user returns. See SECRT-2254/2220.
  useEffect(() => {
    return () => {
      resetArtifactPanel();
    };
  }, [resetArtifactPanel]);
}
