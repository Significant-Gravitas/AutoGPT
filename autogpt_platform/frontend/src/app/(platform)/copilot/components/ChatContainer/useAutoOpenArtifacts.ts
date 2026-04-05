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

interface Props {
  messages: UIMessage<unknown, UIDataTypes, UITools>[];
  sessionId: string | null;
}

export function useAutoOpenArtifacts({ messages, sessionId }: Props) {
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

    const nextFingerprints = new Map<string, string>();
    let nextArtifact: ArtifactRef | null = null;

    for (const message of messages) {
      if (message.role !== "assistant") continue;

      const artifacts = getMessageArtifacts(message);
      const fingerprint = fingerprintArtifacts(artifacts);
      nextFingerprints.set(message.id, fingerprint);

      if (!hasInitializedRef.current || fingerprint.length === 0) {
        continue;
      }

      const previousFingerprint =
        messageFingerprintsRef.current.get(message.id) ?? "";

      if (previousFingerprint === fingerprint) {
        continue;
      }

      nextArtifact = artifacts[artifacts.length - 1] ?? nextArtifact;
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
