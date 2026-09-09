"use client";

import { useListWorkspaceFiles } from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { ListFilesResponse } from "@/app/api/__generated__/models/listFilesResponse";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { useMemo } from "react";
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

  // The chip and the artifacts button both read this on every render, and
  // `messageSnapshots` is rewritten with a fresh array per streamed token —
  // so without memoising, scanning every message part (and compiling a
  // RegExp per matched workspace URI) would run at token cadence.
  const { uploaded, generated, files } = useMemo(() => {
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

    return {
      files,
      uploaded: files.filter((f) => isUploadedFile(f.item)),
      generated: files.filter((f) => !isUploadedFile(f.item)),
    };
  }, [messages, query.data]);

  return {
    uploaded,
    generated,
    isLoading: query.isLoading && !!sessionId,
    isError: query.isError,
    error: query.error,
    isEmpty: !!sessionId && files.length === 0,
  };
}
