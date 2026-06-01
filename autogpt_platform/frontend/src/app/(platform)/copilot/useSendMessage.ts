import { toast } from "@/components/molecules/Toast/use-toast";
import { uploadFileDirect } from "@/lib/direct-upload";
import type { UseChatHelpers } from "@ai-sdk/react";
import type { FileUIPart, UIMessage } from "ai";
import { useEffect, useRef, useState } from "react";
import { useCopilotStreamStore } from "./copilotStreamStore";
import {
  buildWorkspaceFilePart,
  workspaceFileDownloadUrl,
  type WorkspaceAttachment,
} from "./helpers/workspaceAttachments";

const MAX_FILES = 10;
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
  createSession: () => Promise<string | undefined>;
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
  const [isUploadingFiles, setIsUploadingFiles] = useState(false);
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
      sendMessage({
        text,
        files: prebuiltParts.length > 0 ? prebuiltParts : undefined,
      });
      return;
    }
    setIsUploadingFiles(true);
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
        if (prebuiltParts.length > 0) {
          sendMessage({ text, files: prebuiltParts });
          return;
        }
        throw new Error("All file uploads failed");
      }
      // Merge already-stored workspace parts with the freshly uploaded ones so
      // a single message can mix both kinds of attachment.
      const allParts = [...prebuiltParts, ...buildFileParts(uploaded)];
      sendMessage({
        text,
        files: allParts.length > 0 ? allParts : undefined,
      });
    } finally {
      setIsUploadingFiles(false);
    }
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
      .takePendingFirstSend();
    if (!send) return;
    void dispatchRef.current(sessionId, send.text, send.files, parts);
  }, [sessionId]);

  async function onSend(
    message: string,
    files?: File[],
    workspaceFiles?: WorkspaceAttachment[],
  ) {
    const trimmed = message.trim();
    const hasWorkspaceFiles = !!workspaceFiles && workspaceFiles.length > 0;
    if (!trimmed && (!files || files.length === 0) && !hasWorkspaceFiles)
      return;

    if (files && files.length > 0) {
      if (files.length > MAX_FILES) {
        toast({
          title: "Too many files",
          description: `You can attach up to ${MAX_FILES} files at once.`,
          variant: "destructive",
        });
        return;
      }
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
      await dispatchToSession(sessionId, trimmed, files ?? [], [
        ...pendingFileParts,
        ...workspaceParts,
      ]);
      return;
    }

    if (isCreatingSessionRef.current) return;
    isCreatingSessionRef.current = true;
    // Workspace parts must reach the post-creation flush, which reads them
    // from the store via `takePendingFirstSend`. Append so a pre-set part
    // (e.g. workflow-import) isn't clobbered.
    if (workspaceParts.length > 0) {
      const store = useCopilotStreamStore.getState();
      store.setPendingFileParts([...store.pendingFileParts, ...workspaceParts]);
    }
    useCopilotStreamStore
      .getState()
      .setPendingFirstSend({ text: trimmed, files: files ?? [] });
    try {
      await createSession();
    } catch (err) {
      const { setPendingFirstSend, setPendingFileParts } =
        useCopilotStreamStore.getState();
      setPendingFirstSend(null);
      setPendingFileParts([]);
      throw err;
    } finally {
      isCreatingSessionRef.current = false;
    }
  }

  function setPendingFileParts(parts: FileUIPart[]) {
    useCopilotStreamStore.getState().setPendingFileParts(parts);
  }

  return { onSend, isUploadingFiles, setPendingFileParts };
}
