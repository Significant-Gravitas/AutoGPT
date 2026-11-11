import { router } from "expo-router";

import { chatKeys, createSession } from "@/lib/api/chat";
import { setPendingFirstMessage } from "@/features/chat/pendingFirstMessage";
import { queryClient } from "@/lib/session";

export async function startChat(
  text: string,
  expertID?: string,
  files: { id: string; name: string }[] = [],
) {
  const { id } = await createSession(expertID);
  setPendingFirstMessage(id, { text, files });
  void queryClient.invalidateQueries({ queryKey: chatKeys.sessions });
  router.push(`/chat/${id}`);
}
