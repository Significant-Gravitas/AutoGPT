"use client";

import { useEffect, useMemo, useRef } from "react";
import { UIDataTypes, UIMessage, UITools } from "ai";
import { getMostRecentArtifact } from "@/app/(platform)/copilot/components/ChatMessagesContainer/helpers";
import { useCopilotUIStore } from "@/app/(platform)/copilot/store";

interface Props {
  token: string;
  uiMessages: UIMessage<unknown, UIDataTypes, UITools>[];
  isLoading: boolean;
  filePattern: RegExp;
  fileUrlBuilder: (fileId: string) => string;
}

export function useSharedChatArtifacts({
  token,
  uiMessages,
  isLoading,
  filePattern,
  fileUrlBuilder,
}: Props) {
  const isArtifactPanelOpen = useCopilotUIStore((s) => s.artifactPanel.isOpen);
  const openArtifact = useCopilotUIStore((s) => s.openArtifact);
  const resetArtifactPanel = useCopilotUIStore((s) => s.resetArtifactPanel);
  const autoOpenedArtifactIdRef = useRef<string | null>(null);

  const latestArtifact = useMemo(
    () =>
      getMostRecentArtifact(uiMessages, {
        filePattern,
        fileUrlBuilder,
        origin: "agent",
      }),
    [uiMessages, filePattern, fileUrlBuilder],
  );

  useEffect(() => {
    resetArtifactPanel();
    autoOpenedArtifactIdRef.current = null;
    return () => resetArtifactPanel();
  }, [token, resetArtifactPanel]);

  useEffect(() => {
    if (isLoading || !latestArtifact) return;
    if (autoOpenedArtifactIdRef.current === latestArtifact.id) return;
    autoOpenedArtifactIdRef.current = latestArtifact.id;
    openArtifact(latestArtifact);
  }, [isLoading, latestArtifact, openArtifact]);

  return { isArtifactPanelOpen };
}
