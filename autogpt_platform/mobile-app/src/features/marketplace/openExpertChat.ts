import { chatKeys, createSession, listSessions } from "@/lib/api/chat";
import { queryClient } from "@/lib/session";

/** The expert's most recent chat, or a new one when they have none. */
export async function latestOrNewSession(expertID: string) {
  const [latest] = await listSessions({
    expertID,
    limit: 1,
    pinnedFirst: false,
  });
  if (latest) return latest.id;
  const { id } = await createSession(expertID);
  void queryClient.invalidateQueries({ queryKey: chatKeys.sessions });
  return id;
}
