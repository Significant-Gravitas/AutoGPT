"use client";

import { UIDataTypes, UIMessage, UITools } from "ai";
import { useEffect } from "react";
import { useCopilotUIStore } from "../../store";
import { getLatestTurnOverwrites } from "./helpers";

export function useFollowOverwrittenArtifact(
  messages: UIMessage<unknown, UIDataTypes, UITools>[],
) {
  const openFileId = useCopilotUIStore(
    (s) => s.artifactPanel.activeArtifact?.id,
  );
  const followArtifactOverwrite = useCopilotUIStore(
    (s) => s.followArtifactOverwrite,
  );

  useEffect(() => {
    if (!openFileId) return;
    for (const { replacedFileId, next } of getLatestTurnOverwrites(messages)) {
      followArtifactOverwrite(replacedFileId, next);
    }
  }, [messages, openFileId, followArtifactOverwrite]);
}
