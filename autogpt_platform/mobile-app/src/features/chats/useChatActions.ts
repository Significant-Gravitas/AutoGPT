import { router } from "expo-router";
import { useState } from "react";
import { Alert } from "react-native";

import { useToast } from "@/components/useToast";
import { chatTitle } from "@/features/drawer/helpers";
import { chooseAction } from "@/lib/actionMenu";
import {
  chatKeys,
  deleteSession,
  renameSession,
  setSessionPinned,
  type SessionList,
  type SessionSummary,
} from "@/lib/api/chat";
import { tapMenu } from "@/lib/haptics";
import { queryClient } from "@/lib/session";

import {
  chatActionOptions,
  validateTitle,
  withoutSession,
  withPinned,
  withRenamed,
} from "./chatActions";

interface Args {
  activeSessionID?: string;
}

/** The long-press menu on a chat: rename, pin or delete it. */
export function useChatActions({ activeSessionID }: Args) {
  const [renameTarget, setRenameTarget] = useState<SessionSummary | null>(null);
  const { toast, show } = useToast();

  async function openMenu(session: SessionSummary) {
    tapMenu();
    const action = await chooseAction(chatTitle(session), chatActionOptions(session));
    if (action === "rename") setRenameTarget(session);
    if (action === "pin") void togglePin(session);
    if (action === "delete") confirmDelete(session);
  }

  function closeRename() {
    setRenameTarget(null);
  }

  function submitRename(text: string) {
    if (!renameTarget) return null;
    const result = validateTitle(text);
    if ("error" in result) return result.error;
    const sessionID = renameTarget.id;
    setRenameTarget(null);
    void mutateList(
      (list) => withRenamed(list, sessionID, result.title),
      () => renameSession(sessionID, result.title),
      "Couldn't update this chat. Try again.",
    );
    return null;
  }

  function togglePin(session: SessionSummary) {
    const isPinned = !session.is_pinned;
    return mutateList(
      (list) => withPinned(list, session.id, isPinned),
      () => setSessionPinned(session.id, isPinned),
      "Couldn't update this chat. Try again.",
    );
  }

  function confirmDelete(session: SessionSummary) {
    Alert.alert("Delete this chat?", "Its messages are deleted for good.", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Delete",
        style: "destructive",
        onPress: () => void removeChat(session.id),
      },
    ]);
  }

  async function removeChat(sessionID: string) {
    const isDeleted = await mutateList(
      (list) => withoutSession(list, sessionID),
      () => deleteSession(sessionID),
      "Couldn't delete this chat. Try again.",
    );
    if (!isDeleted) return;
    if (sessionID === activeSessionID) router.navigate("/");
    queryClient.removeQueries({ queryKey: chatKeys.session(sessionID) });
  }

  // The list changes at once; a failed request puts it back as it was.
  async function mutateList(
    patch: (list: SessionList) => SessionList,
    request: () => Promise<unknown>,
    errorMessage: string,
  ) {
    await queryClient.cancelQueries({ queryKey: chatKeys.sessions });
    const previous = queryClient.getQueryData<SessionList>(chatKeys.sessions);
    queryClient.setQueryData<SessionList>(chatKeys.sessions, (list) =>
      list ? patch(list) : list,
    );
    try {
      await request();
      return true;
    } catch {
      queryClient.setQueryData(chatKeys.sessions, previous);
      show({ message: errorMessage, tone: "error" });
      return false;
    } finally {
      void queryClient.invalidateQueries({ queryKey: chatKeys.sessions });
    }
  }

  return { openMenu, renameTarget, closeRename, submitRename, toast };
}
