import type { UIMessage } from "ai";

import type { DreamOperationsSnapshot } from "@/app/api/__generated__/models/dreamOperationsSnapshot";

/**
 * AI SDK v5 wire type for the dream-pass snapshot event emitted from
 * the backend ``dream_events.py``. Matches ``ResponseType.DREAM_OPERATIONS``
 * on the backend (``data-dream-operations``).
 *
 * Real consumer (rendering a "dream summary" artifact inline in the
 * chat stream) lands in P6 (surface dreams in chat UI) + P9 (daydreaming).
 * Until then we just recognise the part so the UI filters it out as
 * bookkeeping instead of crashing on an unknown part type.
 */
export const DREAM_OPERATIONS_PART_TYPE = "data-dream-operations" as const;

export interface DreamOperationsPartData {
  snapshot: DreamOperationsSnapshot;
  dream_pass_id: string;
  user_id: string;
}

interface DreamOperationsPart {
  type: typeof DREAM_OPERATIONS_PART_TYPE;
  data: DreamOperationsPartData;
}

/**
 * Type guard for the dream-operations part. Mirrors the inline
 * ``part.type === "data-status"`` checks already used elsewhere in the
 * copilot tree (see ``helpers.ts:getLatestAssistantStatusMessage``).
 */
export function isDreamOperationsPart(
  part: UIMessage["parts"][number],
): part is DreamOperationsPart {
  if (part.type !== DREAM_OPERATIONS_PART_TYPE) return false;
  const data = (part as { data?: unknown }).data;
  return !!data && typeof data === "object";
}

/**
 * Pull the typed payload out of a dream-operations part if present.
 *
 * Returns ``null`` for malformed parts (missing ``data``, wrong shape)
 * so the dispatcher can log + ignore without throwing — the AI SDK
 * parser passes through unknown data parts verbatim and we don't want a
 * single bad event to break the whole stream.
 */
export function readDreamOperationsPart(
  part: UIMessage["parts"][number],
): DreamOperationsPartData | null {
  if (part.type !== DREAM_OPERATIONS_PART_TYPE) return null;
  const data = (part as { data?: unknown }).data;
  if (!data || typeof data !== "object") return null;
  const candidate = data as Record<string, unknown>;
  if (typeof candidate.dream_pass_id !== "string") return null;
  if (typeof candidate.user_id !== "string") return null;
  const snapshot = candidate.snapshot;
  if (!snapshot || typeof snapshot !== "object") return null;
  return {
    snapshot: snapshot as DreamOperationsSnapshot,
    dream_pass_id: candidate.dream_pass_id,
    user_id: candidate.user_id,
  };
}

/**
 * Dispatcher hook for the new event variant.
 *
 * Until the P6 chat-surfacing UI lands we just log the snapshot — the
 * point of this stub is to give the AI SDK parser a recognised handler
 * so an unexpected event arriving on a live stream doesn't crash any
 * downstream filter that assumes the part-type union is closed.
 */
export function handleDreamOperationsPart(
  part: UIMessage["parts"][number],
): void {
  const data = readDreamOperationsPart(part);
  if (!data) return;
  console.debug(
    "[copilot] dream.operations event received",
    data.dream_pass_id,
  );
}
