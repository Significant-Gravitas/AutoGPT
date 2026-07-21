"use client";

import { useListWorkspaceFiles } from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { ListFilesResponse } from "@/app/api/__generated__/models/listFilesResponse";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { useCopilotStreamStore } from "../../../../copilotStreamStore";
import { getMessageArtifacts } from "../../../ChatMessagesContainer/helpers";
import { isUploadedFile } from "./helpers";

export interface SessionFile {
  item: WorkspaceFileItem;
  messageID: string | null;
}

export function useSessionFiles(sessionId: string | null) {
  const messages = useCopilotStreamStore((s) =>
    sessionId ? s.messageSnapshots[sessionId] : undefined,
  );

  const query = useListWorkspaceFiles(
    { session_id: sessionId ?? undefined },
    {
      query: {
        enabled: !!sessionId,
        select: (res) => res.data as ListFilesResponse,
      },
    },
  );

  const fileIdToMessageId = new Map<string, string>();
  for (const message of messages ?? []) {
    for (const artifact of getMessageArtifacts(message)) {
      if (!fileIdToMessageId.has(artifact.id)) {
        fileIdToMessageId.set(artifact.id, message.id);
      }
    }
  }

  const files: SessionFile[] = (query.data?.files ?? []).map((item) => ({
    item,
    messageID: fileIdToMessageId.get(item.id) ?? null,
  }));

  const uploaded = files.filter((f) => isUploadedFile(f.item));
  const generated = files.filter((f) => !isUploadedFile(f.item));

  return {
    uploaded,
    generated,
    isLoading: query.isLoading && !!sessionId,
    isError: query.isError,
    error: query.error,
    isEmpty: !!sessionId && files.length === 0,
  };
}
