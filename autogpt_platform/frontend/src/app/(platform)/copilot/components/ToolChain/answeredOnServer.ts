import { getV2GetSession } from "@/app/api/__generated__/endpoints/chat/chat";
import { answeredIdsInRows } from "../../cardDecisions";

const PAGE_SIZE = 100;
// Past this the card sends as it would; the chat route still refuses a
// second answer.
const MAX_PAGES = 20;

/** The cards the stored chat already answered, so a tab opened before another
 *  one answered sends only what is still open. An answer always comes after
 *  its card, so this reads back from the newest row until every card in
 *  `confirmationIds` is in view. Null when the chat could not be read: the
 *  card then sends as it would, and the chat route still refuses a second
 *  answer; this only spares the stale tab from showing one. */
export async function answeredIdsOnServer(
  sessionId: string,
  confirmationIds: string[],
): Promise<Set<string> | null> {
  const rows: unknown[] = [];
  let beforeSequence: number | undefined;
  try {
    for (let page = 0; page < MAX_PAGES; page++) {
      const response = await getV2GetSession(sessionId, {
        limit: PAGE_SIZE,
        before_sequence: beforeSequence,
      });
      if (response.status !== 200) return null;
      rows.push(...response.data.messages);
      const { has_more_messages, oldest_sequence } = response.data;
      if (!has_more_messages || oldest_sequence == null) break;
      if (confirmationIds.every((id) => mentions(rows, id))) break;
      beforeSequence = oldest_sequence;
    }
  } catch {
    return null;
  }
  return answeredIdsInRows(rows);
}

// The card's own row names its confirmation id, and is older than any answer.
function mentions(rows: unknown[], confirmationId: string) {
  return rows.some((row) => {
    if (!row || typeof row !== "object") return false;
    const { content } = row as Record<string, unknown>;
    return typeof content === "string" && content.includes(confirmationId);
  });
}
