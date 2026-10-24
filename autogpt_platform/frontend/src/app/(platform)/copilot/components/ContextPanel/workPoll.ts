import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import { convertChatSessionMessagesToUiMessages } from "../../helpers/convertChatSessionToUiMessages";
import {
  type ChatDelegation,
  type LiveDelegationStatus,
  getChatDelegations,
} from "../../delegations";

export const WORK_POLL_MS = 5000;
/** Like the sub-session poll: a panel left open stops refreshing the chat
 *  after this long without the user touching it. */
export const WORK_POLL_CAP_MS = 5 * 60_000;

const LIVE = new Set<LiveDelegationStatus>([
  "running",
  "queued",
  "needs-input",
]);

// Keyed by the response object React Query hands out, so the transcript is
// converted once per fetch rather than on every render and interval tick.
const delegationCache = new WeakMap<SessionDetailResponse, ChatDelegation[]>();

export function delegationsOf(
  session: SessionDetailResponse,
): ChatDelegation[] {
  const cached = delegationCache.get(session);
  if (cached) return cached;
  const delegations = getChatDelegations(
    convertChatSessionMessagesToUiMessages(
      session.id,
      session.messages ?? [],
      // A turn still streaming has calls with no result yet; marking them
      // complete would read a hand-off in progress as stopped.
      { isComplete: !session.active_stream },
    ).messages,
  );
  delegationCache.set(session, delegations);
  return delegations;
}

interface PollInputs {
  session: SessionDetailResponse | null;
  liveStatuses: Record<string, LiveDelegationStatus>;
  armedAt: number;
  now: number;
}

/** Refresh the chat while its own turn streams or a teammate is live. A
 *  hand-off counts only by the status its probe read off the teammate's
 *  session: a transcript frozen at "running" (Otto never polled it again)
 *  would otherwise keep this going forever. */
export function workPollInterval({
  session,
  liveStatuses,
  armedAt,
  now,
}: PollInputs): number | false {
  if (!session || now - armedAt > WORK_POLL_CAP_MS) return false;
  if (session.active_stream) return WORK_POLL_MS;
  const live = delegationsOf(session).some((delegation) => {
    const status = liveStatuses[delegation.toolCallId];
    return status !== undefined && LIVE.has(status);
  });
  return live ? WORK_POLL_MS : false;
}
