import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { OPENUI_MIME } from "@/lib/openui/file";
import type { ArtifactRef } from "../../store";
import { workspaceFileDownloadUrl } from "../../helpers/workspaceAttachments";

export const MISSING_ORIGIN_NOTE =
  "The chat this view came from no longer exists.";

export type SubmitRoute =
  | { kind: "none" }
  | { kind: "here" }
  | { kind: "chat"; sessionId: string; navigate: boolean };

interface RouteArgs {
  originSessionId: string | null;
  currentSessionId: string | null;
  canSendHere: boolean;
}

export function resolveSubmitRoute({
  originSessionId,
  currentSessionId,
  canSendHere,
}: RouteArgs): SubmitRoute {
  if (!originSessionId) return { kind: "none" };
  if (originSessionId !== currentSessionId)
    return { kind: "chat", sessionId: originSessionId, navigate: true };
  if (canSendHere) return { kind: "here" };
  return { kind: "chat", sessionId: originSessionId, navigate: false };
}

export function isNotFound(error: unknown) {
  return error instanceof ApiError && error.status === 404;
}

export function openUIArtifactRef(fileId: string, title: string): ArtifactRef {
  return {
    id: fileId,
    title,
    mimeType: OPENUI_MIME,
    sourceUrl: workspaceFileDownloadUrl(fileId),
    origin: "agent",
  };
}
