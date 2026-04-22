import { toast } from "@/components/molecules/Toast/use-toast";
import { uploadFileDirect } from "@/lib/direct-upload";
import type { UseChatHelpers } from "@ai-sdk/react";
import type { FileUIPart, UIMessage } from "ai";
import { useEffect, useRef, useState } from "react";

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

// Module-scope so the queued send survives the CopilotChatHost remount that
// fires when sessionId transitions from null to the freshly-created id. Per-
// instance React refs would be wiped in that window — the "new"-keyed host
// unmounts after createSession resolves and the "<id>"-keyed host mounts with
// fresh refs, so a drain effect inside the hook would see no queue.
let queuedFirstSend: { text: string; files: File[] } | null = null;
let pendingFileParts: FileUIPart[] = [];

/**
 * Orchestrates send-message flow: validates input, uploads attached files,
 * creates a session if one doesn't exist yet, and dispatches the message
 * once the session is ready.
 *
 * The "wait for session creation then send" path is implemented via a
 * queued module variable + effect rather than promise chaining because
 * `sendMessage` closes over the stream's `sessionId` via `useChat` — we need
 * React to re-render with the new sessionId before the bound transport can
 * send.
 */
export function useSendMessage({
  sessionId,
  sendMessage,
  createSession,
  isUserStoppingRef,
}: Args) {
  const [isUploadingFiles, setIsUploadingFiles] = useState(false);

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
      url: `/api/proxy/api/workspace/files/${f.file_id}/download`,
    }));
  }

  async function dispatchToSession(
    sid: string,
    text: string,
    files: File[],
    prebuiltParts: FileUIPart[],
  ) {
    if (prebuiltParts.length > 0) {
      sendMessage({ text, files: prebuiltParts });
      return;
    }
    if (files.length === 0) {
      sendMessage({ text });
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
        throw new Error("All file uploads failed");
      }
      const fileParts = buildFileParts(uploaded);
      sendMessage({
        text,
        files: fileParts.length > 0 ? fileParts : undefined,
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
    if (!sessionId || !queuedFirstSend) return;
    const queued = queuedFirstSend;
    queuedFirstSend = null;
    const prebuiltParts = pendingFileParts;
    pendingFileParts = [];
    void dispatchRef.current(
      sessionId,
      queued.text,
      queued.files,
      prebuiltParts,
    );
  }, [sessionId]);

  async function onSend(message: string, files?: File[]) {
    const trimmed = message.trim();
    if (!trimmed && (!files || files.length === 0)) return;

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

    if (sessionId) {
      const prebuiltParts = pendingFileParts;
      pendingFileParts = [];
      await dispatchToSession(sessionId, trimmed, files ?? [], prebuiltParts);
      return;
    }

    queuedFirstSend = { text: trimmed, files: files ?? [] };
    try {
      await createSession();
    } catch (err) {
      queuedFirstSend = null;
      pendingFileParts = [];
      throw err;
    }
  }

  function setPendingFileParts(parts: FileUIPart[]) {
    pendingFileParts = parts;
  }

  return { onSend, isUploadingFiles, setPendingFileParts };
}
