import { environment } from "@/services/environment";
import { DefaultChatTransport } from "ai";
import type { FileUIPart } from "ai";

import { useCopilotStreamStore } from "./copilotStreamStore";
import { getCopilotAuthHeaders } from "./helpers";
import type { CopilotLlmModel, CopilotMode } from "./store";

interface CreateTransportArgs {
  sessionId: string;
  /**
   * Ref to the current autopilot mode. Kept as a ref (rather than captured
   * value) so mid-session mode changes are picked up on the next send without
   * recreating the transport — recreating would reset `useChat`'s internal
   * `Chat` instance and break mid-session streaming.
   */
  copilotModeRef: React.MutableRefObject<CopilotMode | undefined>;
  /** Ref to the current model tier. See `copilotModeRef` for rationale. */
  copilotModelRef: React.MutableRefObject<CopilotLlmModel | undefined>;
}

/**
 * Build the `DefaultChatTransport` that wires `useChat` directly at the
 * Python backend's SSE endpoint (bypassing the Next.js serverless proxy to
 * avoid the Vercel 800 s function timeout on long-running tasks).
 *
 * Two closures are attached:
 *  - `prepareSendMessagesRequest` — POST new user turns (includes file_ids,
 *    mode, model).
 *  - `prepareReconnectToStreamRequest` — GET-resume existing turns, appending
 *    `?last_chunk_id=…` from the per-session coord so the backend's XREAD can
 *    start at the exclusive successor instead of replaying from zero.
 */
export function createCopilotTransport({
  sessionId,
  copilotModeRef,
  copilotModelRef,
}: CreateTransportArgs) {
  const baseUrl = `${environment.getAGPTServerBaseUrl()}/api/chat/sessions/${sessionId}/stream`;

  return new DefaultChatTransport({
    api: baseUrl,
    prepareSendMessagesRequest: async ({ messages }) => {
      const last = messages[messages.length - 1];
      // Extract file_ids from FileUIPart entries on the message
      const fileIds = last.parts
        ?.filter((p): p is FileUIPart => p.type === "file")
        .map((p) => {
          // URL is like /api/proxy/api/workspace/files/{id}/download
          const match = p.url.match(/\/workspace\/files\/([^/]+)\//);
          return match?.[1];
        })
        .filter(Boolean) as string[] | undefined;
      return {
        body: {
          message: (
            last.parts?.map((p) => (p.type === "text" ? p.text : "")) ?? []
          ).join(""),
          is_user_message: last.role === "user",
          context: null,
          file_ids: fileIds && fileIds.length > 0 ? fileIds : null,
          mode: copilotModeRef.current ?? null,
          model: copilotModelRef.current ?? null,
        },
        headers: await getCopilotAuthHeaders(),
      };
    },
    prepareReconnectToStreamRequest: async () => {
      const coord = useCopilotStreamStore.getState().getCoord(sessionId);
      const cursor = coord.lastChunkId;
      return {
        api: cursor
          ? `${baseUrl}?last_chunk_id=${encodeURIComponent(cursor)}`
          : baseUrl,
        headers: await getCopilotAuthHeaders(),
      };
    },
  });
}
