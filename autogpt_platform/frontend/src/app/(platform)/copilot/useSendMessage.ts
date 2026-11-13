import { useEffect, useInsertionEffect, useRef } from "react";
import { pendingUploadKey, useCopilotStreamStore } from "./copilotStreamStore";
import { toastSendFailure } from "./helpers/sendFailureToast";
import type { ExpertKickoffMetadata } from "./expertKickoff";
import {
  describePendingUpload,
  hasAttachments,
  sendWithAttachments,
  validateAttachments,
  type OutgoingMessage,
} from "./helpers/sendWithAttachments";
import {
  buildStoredAttachmentParts,
  type StoredAttachmentPart,
  type WorkspaceAttachment,
} from "./helpers/workspaceAttachments";
import { useCopilotUIStore } from "./store";
import type { SendInput } from "./stream/turnRuntime";

type SendMessageFn = (message: SendInput) => Promise<void>;

interface Args {
  sessionId: string | null;
  sendMessage: SendMessageFn;
  createSession: (options?: {
    expertKickoff?: boolean;
  }) => Promise<string | undefined>;
  isUserStoppingRef: React.MutableRefObject<boolean>;
}

/**
 * Orchestrates send-message flow: validates input, uploads attached files,
 * creates a session if one doesn't exist yet, and dispatches the message
 * once the session is ready.
 *
 * The "wait for session creation then send" path uses a slot on the Zustand
 * stream store (rather than React refs) because `CopilotPage` keys the chat
 * subtree by sessionId — the moment a session is created the `"new"`-keyed
 * host unmounts and the `"<id>"`-keyed one mounts with fresh refs. The store
 * slot survives that remount so the new host can pick up the pending send.
 */
export function useSendMessage({
  sessionId,
  sendMessage,
  createSession,
  isUserStoppingRef,
}: Args) {
  // Synchronous concurrency guard for the "no session yet" path: rapid
  // double-press / double-Enter would otherwise overwrite `pendingFirstSend`
  // (losing the first message) AND fire two parallel `createSession`
  // requests (creating duplicate sessions). The ref flips before the
  // mutation dispatches and resets in `finally`, so a second call inside
  // the same tick short-circuits.
  const isCreatingSessionRef = useRef(false);

  function dispatchToSession(sid: string, message: OutgoingMessage) {
    return sendWithAttachments({ sessionId: sid, message, sendMessage });
  }

  // Hold dispatchToSession in a ref so the queued-send effect can fire
  // exclusively on sessionId change (the real trigger) while still calling
  // the latest closure — which captures the refreshed `sendMessage` after
  // the session has updated.
  const dispatchRef = useRef(dispatchToSession);
  useInsertionEffect(() => {
    dispatchRef.current = dispatchToSession;
  });

  useEffect(() => {
    if (!sessionId) return;
    const { send, parts } = useCopilotStreamStore
      .getState()
      .takePendingFirstSend(sessionId);
    if (!send) return;
    // `onSend` resolved the moment the session was created, so the composer's
    // own catch (restore draft + chips, toast) can no longer observe this
    // dispatch failing. Recover here instead, or an all-uploads-failed first
    // send rejects with nobody listening and the user loses their message.
    void dispatchRef
      .current(sessionId, {
        text: send.text,
        files: send.files,
        parts,
        metadata: send.metadata,
      })
      .catch((error: unknown) => {
        recoverFailedDeferredSend(send.text, parts, error);
      });
  }, [sessionId]);

  async function onSend(
    message: string,
    files?: File[],
    workspaceFiles?: WorkspaceAttachment[],
    metadata?: ExpertKickoffMetadata,
  ) {
    const trimmed = message.trim();
    const workspaceParts = buildStoredAttachmentParts(workspaceFiles ?? []);
    const outgoing: OutgoingMessage = {
      text: trimmed,
      files: files ?? [],
      parts: workspaceParts,
      metadata,
    };
    if (!trimmed && !hasAttachments(outgoing)) return;
    // Backstop: the composer caps each attach, so the UI cannot reach this.
    if (!validateAttachments(outgoing)) return;

    isUserStoppingRef.current = false;

    if (sessionId) {
      const { pendingFileParts, setPendingFileParts } =
        useCopilotStreamStore.getState();
      setPendingFileParts([]);
      await dispatchToSession(sessionId, {
        ...outgoing,
        parts: [...pendingFileParts, ...workspaceParts],
      });
      return sessionId;
    }

    if (isCreatingSessionRef.current) return;
    isCreatingSessionRef.current = true;
    if (files && files.length > 0) {
      useCopilotStreamStore
        .getState()
        .setPendingUploadSend(null, describePendingUpload(outgoing));
    }
    // Workspace parts must reach the post-creation flush, which reads them
    // from the store via `takePendingFirstSend`. Append so a pre-set part
    // (e.g. workflow-import) isn't clobbered.
    if (workspaceParts.length > 0) {
      const store = useCopilotStreamStore.getState();
      store.setPendingFileParts([...store.pendingFileParts, ...workspaceParts]);
    }
    useCopilotStreamStore
      .getState()
      .setPendingFirstSend({ text: trimmed, files: files ?? [], metadata });
    return await createFirstSendSession(
      metadata?.kind === "expert_kickoff",
    ).finally(() => {
      isCreatingSessionRef.current = false;
    });
  }

  async function createFirstSendSession(expertKickoff: boolean) {
    try {
      return await createSession({ expertKickoff });
    } catch (err) {
      const {
        pendingFirstSendSessionId,
        setPendingFirstSend,
        setPendingFileParts,
        clearPendingUploadSend,
      } = useCopilotStreamStore.getState();
      setPendingFirstSend(null);
      setPendingFileParts([]);
      // `createSession` can fail after binding the send to the new id, which
      // moves the placeholder off the unbound key. Clear both keys so an
      // aborted first send never strands one on the freshly created session.
      clearPendingUploadSend(null);
      if (pendingFirstSendSessionId)
        clearPendingUploadSend(pendingFirstSendSessionId);
      throw err;
    }
  }

  function setPendingFileParts(parts: StoredAttachmentPart[]) {
    useCopilotStreamStore.getState().setPendingFileParts(parts);
  }

  // A placeholder belongs to the chat it was sent from: one owned by another
  // session (the user switched threads mid-upload) must not show up here,
  // and the not-yet-bound first send only shows in the new-chat host.
  const pendingSend = useCopilotStreamStore(
    (s) => s.pendingUploadSends[pendingUploadKey(sessionId)] ?? null,
  );
  // Derived from the store rather than local state so the composer stays
  // locked when the host remounts while this chat's upload is in flight.
  const isUploadingFiles = pendingSend !== null;

  return { onSend, isUploadingFiles, pendingSend, setPendingFileParts };
}

/**
 * Failure recovery for a send whose caller can no longer observe it: the
 * first send of a new chat (dispatched from an effect long after `onSend`
 * returned) and a follow-up held until the local stream settled (see
 * `useCopilotPage`), whose `onSend` resolves once the turn is dispatched.
 *
 * Everything goes through the stores because the `"new"`-keyed host that
 * started a first send is already unmounted — its draft state and chips are
 * gone. The text comes back as the composer's initial prompt and the
 * workspace references as pending file parts; local `File` chips cannot be
 * restored, they only ever existed in that unmounted host.
 */
export function recoverFailedDeferredSend(
  text: string,
  parts: StoredAttachmentPart[],
  error: unknown,
) {
  if (text) useCopilotUIStore.getState().setInitialPrompt(text);
  if (parts.length > 0)
    useCopilotStreamStore.getState().setPendingFileParts(parts);
  toastSendFailure({
    title: "Couldn't send message",
    error,
    recovery: text
      ? "your message is back in the composer"
      : "your files were not sent",
  });
}
