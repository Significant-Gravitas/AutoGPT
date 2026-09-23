import { toast } from "@/components/molecules/Toast/use-toast";
import { uploadFileDirect } from "@/lib/direct-upload";
import type { UseChatHelpers } from "@ai-sdk/react";
import type { FileUIPart, UIMessage } from "ai";
import { useEffect, useRef } from "react";
import {
  pendingUploadKey,
  useCopilotStreamStore,
  type PendingUploadAttachment,
  type PendingUploadSend,
} from "./copilotStreamStore";
import { describeSendFailure } from "./components/ChatInput/helpers";
import type { ExpertKickoffMetadata } from "./expertKickoff";
import {
  buildWorkspaceFilePart,
  MAX_ATTACHMENTS,
  workspaceFileDownloadUrl,
  type WorkspaceAttachment,
} from "./helpers/workspaceAttachments";
import { useCopilotUIStore } from "./store";

const MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024;

interface UploadedFile {
  file_id: string;
  name: string;
  mime_type: string;
}

type SendMessageFn = UseChatHelpers<UIMessage>["sendMessage"];

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

  async function uploadFiles(
    files: File[],
    sid: string,
  ): Promise<UploadedFile[]> {
    const results = await Promise.allSettled(
      files.map(async (file) => {
        try {
          const data = await uploadFileDirect(file, sid);
          if (!data.file_id) throw new Error("No file_id returned");
          return {
            file_id: data.file_id,
            name: data.name || file.name,
            mime_type: data.mime_type || "application/octet-stream",
          } as UploadedFile;
        } catch (err) {
          console.error("File upload failed:", err);
          toast({
            title: "File upload failed",
            description: file.name,
            variant: "destructive",
          });
          throw err;
        }
      }),
    );
    return results
      .filter(
        (r): r is PromiseFulfilledResult<UploadedFile> =>
          r.status === "fulfilled",
      )
      .map((r) => r.value);
  }

  function buildFileParts(uploaded: UploadedFile[]): FileUIPart[] {
    return uploaded.map((f) => ({
      type: "file" as const,
      mediaType: f.mime_type,
      filename: f.name,
      url: workspaceFileDownloadUrl(f.file_id),
    }));
  }

  async function dispatchToSession(
    sid: string,
    text: string,
    files: File[],
    prebuiltParts: FileUIPart[],
    metadata?: ExpertKickoffMetadata,
  ) {
    // The per-click UUID that becomes the backend's ``ChatMessage.id``
    // is generated inside the transport's ``prepareSendMessagesRequest``
    // (one call per ``sendMessages``, stable across SDK-internal
    // retries) — NOT here.  AI SDK's ``messageId`` arg means
    // "replace-existing-message" (edit mode), not "id of a new
    // message", so passing it would put the SDK into edit-mode with no
    // target and break the optimistic-render path that pushes the user
    // bubble into ``messages`` synchronously.
    if (files.length === 0) {
      await sendMessage({
        text,
        files: prebuiltParts.length > 0 ? prebuiltParts : undefined,
        metadata,
      });
      return;
    }
    // The bubble shows right away; the transcript reads "Uploading N files…"
    // until the uploads land and the real message takes the placeholder's
    // spot. The slot is re-set here (not only in onSend) because this is the
    // one path every send with files goes through, including the flush after
    // session creation.
    const pending = describePendingUpload(text, files, prebuiltParts);
    useCopilotStreamStore.getState().setPendingUploadSend(sid, pending);
    let send: Promise<void> | undefined;
    try {
      const uploaded = await uploadFiles(files, sid);
      if (uploaded.length === 0) {
        toast({
          title: "File upload failed",
          description: "Could not upload any files. Please try again.",
          variant: "destructive",
        });
        // The workspace references didn't fail to upload (they need no upload),
        // so don't discard them just because the local uploads failed.
        if (prebuiltParts.length === 0) {
          throw new Error("All file uploads failed");
        }
        send = sendMessage({ text, files: prebuiltParts, metadata });
      } else {
        // Merge already-stored workspace parts with the freshly uploaded ones so
        // a single message can mix both kinds of attachment.
        const allParts = [...prebuiltParts, ...buildFileParts(uploaded)];
        send = sendMessage({ text, files: allParts, metadata });
      }
    } finally {
      // `sendMessage` pushes the user bubble into `messages` synchronously,
      // so the placeholder can go in the same tick with no gap between the
      // two. Its promise only settles when the whole stream ends, which is
      // why the cleanup happens here and not after the await below.
      useCopilotStreamStore.getState().clearPendingUploadSend(sid, pending);
    }
    await send;
  }

  // Hold dispatchToSession in a ref so the queued-send effect can fire
  // exclusively on sessionId change (the real trigger) while still calling
  // the latest closure — which captures the refreshed `sendMessage` after
  // the session has updated.
  const dispatchRef = useRef(dispatchToSession);
  dispatchRef.current = dispatchToSession;

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
      .current(sessionId, send.text, send.files, parts, send.metadata)
      .catch((error: unknown) => {
        recoverFailedFirstSend(send.text, parts, error);
      });
  }, [sessionId]);

  async function onSend(
    message: string,
    files?: File[],
    workspaceFiles?: WorkspaceAttachment[],
    metadata?: ExpertKickoffMetadata,
  ) {
    const trimmed = message.trim();
    const hasWorkspaceFiles = !!workspaceFiles && workspaceFiles.length > 0;
    if (!trimmed && (!files || files.length === 0) && !hasWorkspaceFiles)
      return;

    // Backstop: the composer caps each attach, so the UI cannot reach this.
    // Workspace references count too — uploaded or not, every attachment
    // becomes one `file_ids` entry on the request the backend bounds.
    if (
      (files?.length ?? 0) + (workspaceFiles?.length ?? 0) >
      MAX_ATTACHMENTS
    ) {
      toast({
        title: "Too many attachments",
        description: `You can attach up to ${MAX_ATTACHMENTS} attachments per message.`,
        variant: "destructive",
      });
      return;
    }

    if (files && files.length > 0) {
      const oversized = files.filter((f) => f.size > MAX_FILE_SIZE_BYTES);
      if (oversized.length > 0) {
        toast({
          title: "File too large",
          description: `${oversized[0].name} exceeds the 100 MB limit.`,
          variant: "destructive",
        });
        return;
      }
    }

    isUserStoppingRef.current = false;

    const workspaceParts = (workspaceFiles ?? []).map(buildWorkspaceFilePart);

    if (sessionId) {
      const { pendingFileParts, setPendingFileParts } =
        useCopilotStreamStore.getState();
      setPendingFileParts([]);
      await dispatchToSession(
        sessionId,
        trimmed,
        files ?? [],
        [...pendingFileParts, ...workspaceParts],
        metadata,
      );
      return;
    }

    if (isCreatingSessionRef.current) return;
    isCreatingSessionRef.current = true;
    if (files && files.length > 0) {
      useCopilotStreamStore
        .getState()
        .setPendingUploadSend(
          null,
          describePendingUpload(trimmed, files, workspaceParts),
        );
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
    try {
      await createSession({
        expertKickoff: metadata?.kind === "expert_kickoff",
      });
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
    } finally {
      isCreatingSessionRef.current = false;
    }
  }

  function setPendingFileParts(parts: FileUIPart[]) {
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
 * Failure recovery for the first send of a new chat, which is dispatched from
 * an effect long after `onSend` returned.
 *
 * Everything goes through the stores because the `"new"`-keyed host that
 * started the send is already unmounted — its draft state and chips are gone.
 * The text comes back as the composer's initial prompt and the workspace
 * references as pending file parts; local `File` chips cannot be restored,
 * they only ever existed in that unmounted host.
 */
function recoverFailedFirstSend(
  text: string,
  parts: FileUIPart[],
  error: unknown,
) {
  if (text) useCopilotUIStore.getState().setInitialPrompt(text);
  if (parts.length > 0)
    useCopilotStreamStore.getState().setPendingFileParts(parts);
  toast({
    title: "Couldn't send message",
    description: describeSendFailure(
      error,
      text
        ? "your message is back in the composer"
        : "your files were not sent",
    ),
    variant: "destructive",
  });
}

function describePendingUpload(
  text: string,
  files: File[],
  prebuiltParts: FileUIPart[],
): PendingUploadSend {
  const stored: PendingUploadAttachment[] = prebuiltParts.map((part) => ({
    name: part.filename ?? "file",
    mediaType: part.mediaType,
    isUploading: false,
  }));
  const local: PendingUploadAttachment[] = files.map((file) => ({
    name: file.name,
    mediaType: file.type || "application/octet-stream",
    sizeBytes: file.size,
    isUploading: true,
  }));
  return { text, attachments: [...stored, ...local] };
}
