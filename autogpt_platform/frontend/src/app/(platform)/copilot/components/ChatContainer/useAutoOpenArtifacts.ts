"use client";

import { UIDataTypes, UIMessage, UITools } from "ai";
import { useEffect, useRef } from "react";
import type { ArtifactRef } from "../../store";
import { useCopilotUIStore } from "../../store";
import { getMessageArtifacts } from "../ChatMessagesContainer/helpers";

function fingerprintArtifacts(artifacts: ArtifactRef[]): string {
  return artifacts
    .map((a) => `${a.id}:${a.title}:${a.mimeType ?? ""}:${a.sourceUrl}`)
    .join("|");
}

interface UseAutoOpenArtifactsOptions {
  messages: UIMessage<unknown, UIDataTypes, UITools>[];
  sessionId: string | null;
}

export function useAutoOpenArtifacts({
  messages,
  sessionId,
}: UseAutoOpenArtifactsOptions) {
  const openArtifact = useCopilotUIStore((state) => state.openArtifact);
  const messageFingerprintsRef = useRef<Map<string, string>>(new Map());
  const hasInitializedRef = useRef(false);

  useEffect(() => {
    messageFingerprintsRef.current = new Map();
    hasInitializedRef.current = false;
  }, [sessionId]);

  useEffect(() => {
    if (messages.length === 0) {
      messageFingerprintsRef.current = new Map();
      return;
    }

    // Only scan messages whose fingerprint might have changed since the
    // last pass: that's the last assistant message (currently streaming)
    // plus any assistant message whose id isn't in the baseline yet.
    // This keeps the cost O(new+tail), not O(all messages), on every chunk.
    const previous = messageFingerprintsRef.current;
    const nextFingerprints = new Map<string, string>(previous);
    let nextArtifact: ArtifactRef | null = null;
    const lastAssistantIdx = (() => {
      for (let i = messages.length - 1; i >= 0; i--) {
        if (messages[i].role === "assistant") return i;
      }
      return -1;
    })();

    for (let i = 0; i < messages.length; i++) {
      const message = messages[i];
      if (message.role !== "assistant") continue;
      const isTailAssistant = i === lastAssistantIdx;
      const isNewMessage = !previous.has(message.id);
      if (!isTailAssistant && !isNewMessage) continue;

      const artifacts = getMessageArtifacts(message);
      const fingerprint = fingerprintArtifacts(artifacts);
      nextFingerprints.set(message.id, fingerprint);

      if (!hasInitializedRef.current || fingerprint.length === 0) {
        continue;
      }

      const previousFingerprint = previous.get(message.id) ?? "";
      if (previousFingerprint === fingerprint) continue;

      nextArtifact = artifacts[artifacts.length - 1] ?? nextArtifact;
    }

    // Drop entries for messages that no longer exist (e.g. history truncated).
    const liveIds = new Set(messages.map((m) => m.id));
    for (const id of nextFingerprints.keys()) {
      if (!liveIds.has(id)) nextFingerprints.delete(id);
    }

    messageFingerprintsRef.current = nextFingerprints;

    if (!hasInitializedRef.current) {
      hasInitializedRef.current = true;
      return;
    }

    if (nextArtifact) {
      openArtifact(nextArtifact);
    }
  }, [messages, openArtifact]);
}
