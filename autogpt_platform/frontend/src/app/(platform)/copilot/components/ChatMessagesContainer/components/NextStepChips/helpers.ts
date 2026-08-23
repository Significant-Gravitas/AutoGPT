import { readSuggestionsPart } from "../../../../nextStepSuggestions";
import type { MessagePart } from "../../helpers";

/**
 * Collect the chip labels carried by one assistant message.
 *
 * The backend publishes at most one ``data-suggestions`` part per turn,
 * but a resumed stream replays it — the last part wins so a replay never
 * doubles the chip row.
 */
export function getNextStepSuggestions(parts: MessagePart[]): string[] {
  let latest: string[] = [];
  for (const part of parts) {
    const suggestions = readSuggestionsPart(part);
    if (suggestions && suggestions.length > 0) latest = suggestions;
  }
  return latest;
}
