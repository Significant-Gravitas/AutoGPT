import type { UIMessage } from "ai";

/**
 * AI SDK v5 wire type for the next-step chips the model offers at the end
 * of a substantive turn. Matches ``ResponseType.SUGGESTIONS`` on the
 * backend (``data-suggestions``), published by the ``suggest_next_steps``
 * tool in ``copilot/tools/suggest_next_steps.py``.
 */
export const SUGGESTIONS_PART_TYPE = "data-suggestions" as const;

/** Mirror of ``MAX_SUGGESTIONS`` in ``copilot/response_model.py``. */
export const MAX_SUGGESTIONS = 3;

/**
 * Pull the chip labels out of a ``data-suggestions`` part, or ``null`` for
 * any other part. Malformed payloads yield ``null`` rather than throwing —
 * the AI SDK passes unknown data parts through verbatim and one bad event
 * must not take down the message list.
 */
export function readSuggestionsPart(
  part: UIMessage["parts"][number],
): string[] | null {
  if (part.type !== SUGGESTIONS_PART_TYPE) return null;
  const data = (part as { data?: unknown }).data;
  if (!data || typeof data !== "object") return null;
  const raw = (data as Record<string, unknown>).suggestions;
  if (!Array.isArray(raw)) return null;
  return raw
    .filter((entry): entry is string => typeof entry === "string")
    .map((entry) => entry.trim())
    .filter(Boolean)
    .slice(0, MAX_SUGGESTIONS);
}
