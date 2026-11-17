import { chatKeys, createSession, listSessions } from "@/lib/api/chat";
import { queryClient } from "@/lib/session";

import { awaitFirstChat } from "./firstChat";

async function latestSession(expertID: string) {
  const [latest] = await listSessions({ expertID, limit: 1, pinnedFirst: false });
  return latest?.id ?? null;
}

async function newSession(expertID: string) {
  const { id } = await createSession(expertID);
  void queryClient.invalidateQueries({ queryKey: chatKeys.sessions });
  return id;
}

function sleep(ms: number) {
  return new Promise<void>((resolve) => setTimeout(resolve, ms));
}

/** The expert's most recent chat, or a new one when they have none. */
export async function latestOrNewSession(expertID: string) {
  return (await latestSession(expertID)) ?? newSession(expertID);
}

/** For an expert hired moments ago: the chat their hello lands in. */
export function firstChatAfterHire(expertID: string) {
  return awaitFirstChat({
    findLatest: () => latestSession(expertID),
    create: () => newSession(expertID),
    wait: sleep,
  });
}
