import { v4 as uuidv4 } from "uuid";

export interface QueuedMessage {
  /** This tab's own key for the chip. */
  id: string;
  text: string;
  /** The server's id for the queued message, once this tab knows it. */
  serverId?: string | null;
}

export interface ServerItem {
  id: string | null;
  content: string;
}

/** The queue as the server lists it; an older server lists text alone. */
export function serverItemsOf(data: {
  messages: string[];
  items?: { id: string; content: string }[] | null;
}): ServerItem[] {
  if (data.items && data.items.length === data.messages.length)
    return data.items;
  return data.messages.map((content) => ({ id: null, content }));
}

/**
 * This send's chip, unless a peek answered before the send did and already
 * listed it: the server's id says it is the same message.
 */
export function adoptOrAppend(
  prev: QueuedMessage[],
  entry: QueuedMessage,
): QueuedMessage[] {
  if (entry.serverId && prev.some((e) => e.serverId === entry.serverId))
    return prev;
  return [...prev, entry];
}

/**
 * At a turn's start: prepend what the server queues that the strip does not
 * know yet (oldest first). A chip without a server id, a send still on its
 * way or an older server, accounts for one unknown entry, as before ids.
 */
export function withUnknownRestored(
  current: QueuedMessage[],
  items: ServerItem[],
  inFlightIds: ReadonlySet<string>,
): QueuedMessage[] {
  const known = new Set(
    current.flatMap((e) => (e.serverId ? [e.serverId] : [])),
  );
  const unknown = items.filter((item) => !item.id || !known.has(item.id));
  const unmatched = current.filter(
    (e) => inFlightIds.has(e.id) && !e.serverId,
  ).length;
  const restored = unknown
    .slice(0, Math.max(0, unknown.length - unmatched))
    .map(toChip);
  return restored.length > 0 ? [...restored, ...current] : current;
}

/**
 * Idle or on load the server's queue is the truth. A chip added after the
 * GET went out survives unless the server already lists its message, which
 * happens when that send resolved between the GET and its answer.
 */
export function rebasedToServer(
  current: QueuedMessage[],
  items: ServerItem[],
  inFlightIds: ReadonlySet<string>,
): QueuedMessage[] {
  const listed = new Set(items.flatMap((item) => (item.id ? [item.id] : [])));
  const addedSince = current.filter(
    (e) => !inFlightIds.has(e.id) && !(e.serverId && listed.has(e.serverId)),
  );
  return [...items.map(toChip), ...addedSince];
}

function toChip(item: ServerItem): QueuedMessage {
  // Options force uuid's getRandomValues path (plain-HTTP LAN origins).
  return { id: uuidv4({}), text: item.content, serverId: item.id };
}
