import type { UIMessage } from "ai";
import {
  getV2GetSession,
  postV2CancelSessionTask,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import { isHeldCallRow } from "./components/ChatMessagesContainer/heldCallRows";
import { isSessionLive } from "./components/ToolChain/SubSessionLive";
import { type DelegationStatus, getChatDelegations } from "./delegations";

const IN_FLIGHT = new Set<DelegationStatus>(["running", "queued", "proposed"]);

/** Tool calls of the turn Stop ends: everything after the user's last own
 *  message. Rows the server writes for an answered approval are not the
 *  user's, so a hand-off approved mid-turn stays in it. */
function currentTurnCallIds(messages: UIMessage[]): Set<string> {
  const start = messages.findLastIndex(
    (message) => message.role === "user" && !isHeldCallRow(message),
  );
  const ids = new Set<string>();
  for (const message of messages.slice(start + 1)) {
    if (message.role !== "assistant") continue;
    for (const part of message.parts)
      if ("toolCallId" in part && typeof part.toolCallId === "string")
        ids.add(part.toolCallId);
  }
  return ids;
}

/** Hand-offs this turn opened that may still be running. Held results live
 *  on user rows, so the delegations are read off the whole chat. */
export function getStoppableSubSessionIds(messages: UIMessage[]): string[] {
  const turn = currentTurnCallIds(messages);
  const ids = getChatDelegations(messages).flatMap((delegation) =>
    turn.has(delegation.toolCallId) &&
    delegation.subSessionId &&
    IN_FLIGHT.has(delegation.status)
      ? [delegation.subSessionId]
      : [],
  );
  return [...new Set(ids)];
}

/** The transcript's "running" may be stale; only a session that is live
 *  right now is worth stopping. An unreadable one is stopped to be safe. */
async function isStillLive(subSessionId: string): Promise<boolean> {
  try {
    const res = await getV2GetSession(subSessionId);
    return res.status !== 200 || isSessionLive(res.data);
  } catch {
    return true;
  }
}

async function cancelIfLive(subSessionId: string): Promise<boolean> {
  if (!(await isStillLive(subSessionId))) return true;
  const res = await postV2CancelSessionTask(subSessionId);
  return res.status === 200;
}

/** Stop means stop: the teammates this turn handed work to stop too. Each
 *  cancel is independent; one toast covers any that failed. */
export async function cancelSubSessions(subSessionIds: string[]) {
  if (subSessionIds.length === 0) return;
  const results = await Promise.allSettled(subSessionIds.map(cancelIfLive));
  const failed = results.filter(
    (result) => result.status === "rejected" || !result.value,
  ).length;
  if (failed === 0) return;
  toast({
    title:
      failed === 1
        ? "Could not stop one of the experts"
        : `Could not stop ${failed} experts`,
    description: "They may keep working in their own threads.",
    variant: "destructive",
  });
}
