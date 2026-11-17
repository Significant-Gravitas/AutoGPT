import { router } from "expo-router";

import { chatKeys, createSession, emptySession } from "@/lib/api/chat";
import { setPendingFirstMessage } from "@/features/chat/pendingFirstMessage";
import { queryClient, sessionGeneration } from "@/lib/session";

export async function startChat(
  text: string,
  expertID?: string,
  files: { id: string; name: string }[] = [],
) {
  const startedIn = sessionGeneration();
  const { id } = await createSession(expertID);
  // Signed out meanwhile: the cache is already cleared, so nothing goes back into it.
  if (startedIn !== sessionGeneration()) return;
  setPendingFirstMessage(id, { text, files });
  queryClient.setQueryData(chatKeys.session(id), emptySession(id, expertID));
  void queryClient.invalidateQueries({ queryKey: chatKeys.sessions });
  router.push(`/chat/${id}`);
}
