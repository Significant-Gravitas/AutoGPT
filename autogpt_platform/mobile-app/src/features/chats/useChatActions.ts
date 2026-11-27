import { router, useNavigationContainerRef } from "expo-router";
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
import { queryClient, sessionGeneration } from "@/lib/session";

import {
  chatActionOptions,
  restoreSession,
  validateTitle,
  withoutSession,
  withPinned,
  withRenamed,
} from "./chatActions";

const pendingActions = new Map<number, Set<string>>();

/** The long-press menu on a chat: rename, pin or delete it. */
export function useChatActions() {
  const navigation = useNavigationContainerRef();
  const [rename, setRename] = useState<{ chat: SessionSummary; generation: number } | null>(null);
  const { toast, show } = useToast();

  async function openMenu(session: SessionSummary) {
    const startedIn = sessionGeneration();
    if (pendingActions.get(startedIn)?.has(session.id)) return;
    tapMenu();
    const action = await chooseAction(chatTitle(session), chatActionOptions(session));
    if (startedIn !== sessionGeneration()) return;
    if (action === "rename") setRename({ chat: session, generation: startedIn });
    if (action === "pin") void togglePin(session);
    if (action === "delete") confirmDelete(session);
  }

  function closeRename() {
    setRename(null);
  }

  function submitRename(text: string) {
    if (!rename || rename.generation !== sessionGeneration()) return null;
    const result = validateTitle(text);
    if ("error" in result) return result.error;
    const sessionID = rename.chat.id;
    setRename(null);
    void mutateList(
      sessionID,
      (list) => withRenamed(list, sessionID, result.title),
      () => renameSession(sessionID, result.title),
      "Couldn't update this chat. Try again.",
    );
    return null;
  }

  function togglePin(session: SessionSummary) {
    const isPinned = !session.is_pinned;
    return mutateList(
      session.id,
      (list) => withPinned(list, session.id, isPinned),
      () => setSessionPinned(session.id, isPinned),
      "Couldn't update this chat. Try again.",
    );
  }

  function confirmDelete(session: SessionSummary) {
    const startedIn = sessionGeneration();
    Alert.alert("Delete this chat?", "Its messages are deleted for good.", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Delete",
        style: "destructive",
        onPress: () => {
          if (startedIn === sessionGeneration()) void removeChat(session.id);
        },
      },
    ]);
  }

  async function removeChat(sessionID: string) {
    const startedIn = sessionGeneration();
    const isDeleted = await mutateList(
      sessionID,
      (list) => withoutSession(list, sessionID),
      () => deleteSession(sessionID),
      "Couldn't delete this chat. Try again.",
    );
    if (!isDeleted || startedIn !== sessionGeneration()) return;
    const route = navigation.getCurrentRoute() as
      | { name: string; params?: { sessionId?: unknown } }
      | undefined;
    if (
      route?.name === "chat/[sessionId]" &&
      route.params?.sessionId === sessionID
    ) {
      router.navigate("/");
    }
    queryClient.removeQueries({ queryKey: chatKeys.session(sessionID) });
  }

  async function mutateList(
    sessionID: string,
    patch: (list: SessionList) => SessionList,
    request: () => Promise<unknown>,
    errorMessage: string,
  ) {
    const startedIn = sessionGeneration();
    const pending = pendingActions.get(startedIn) ?? new Set<string>();
    if (pending.has(sessionID)) return false;
    pending.add(sessionID);
    pendingActions.set(startedIn, pending);
    let previous: SessionSummary | undefined;
    let optimistic: SessionSummary | undefined;
    try {
      await queryClient.cancelQueries({ queryKey: chatKeys.sessions });
      if (startedIn !== sessionGeneration()) return false;
      const list = queryClient.getQueryData<SessionList>(chatKeys.sessions);
      if (list) {
        previous = list.sessions.find((session) => session.id === sessionID);
        const updated = patch(list);
        optimistic = updated.sessions.find((session) => session.id === sessionID);
        queryClient.setQueryData(chatKeys.sessions, updated);
      }
      await request();
      return true;
    } catch {
      if (startedIn !== sessionGeneration()) return false;
      queryClient.setQueryData<SessionList>(chatKeys.sessions, (list) =>
        list && previous ? restoreSession(list, previous, optimistic) : list,
      );
      show({ message: errorMessage, tone: "error" });
      return false;
    } finally {
      pending.delete(sessionID);
      if (pending.size === 0) {
        pendingActions.delete(startedIn);
        if (startedIn === sessionGeneration()) {
          void queryClient.invalidateQueries({ queryKey: chatKeys.sessions });
        }
      }
    }
  }

  return { openMenu, renameTarget: rename?.chat ?? null, closeRename, submitRename, toast };
}
