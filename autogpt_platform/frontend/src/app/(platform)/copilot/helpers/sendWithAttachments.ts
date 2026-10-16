import { toast } from "@/components/molecules/Toast/use-toast";
import { uploadFileDirect } from "@/lib/direct-upload";
import type { FileUIPart, UIMessage } from "ai";
import {
  useCopilotStreamStore,
  type PendingUploadAttachment,
  type PendingUploadSend,
} from "../copilotStreamStore";
import type { ExpertKickoffMetadata } from "../expertKickoff";
import type { SendInput } from "../stream/turnRuntime";
import {
  isFolderPart,
  MAX_ATTACHMENTS,
  workspaceFileDownloadUrl,
  type StoredAttachmentPart,
} from "./workspaceAttachments";

const MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024;

/** One message on its way out: its text, the local files still to upload,
 *  and the workspace references that are already stored. */
export interface OutgoingMessage {
  text: string;
  files: File[];
  parts: StoredAttachmentPart[];
  metadata?: ExpertKickoffMetadata;
}

interface UploadedFile {
  file_id: string;
  name: string;
  mime_type: string;
}

type SendMessageFn = (message: SendInput) => Promise<void>;

interface SendArgs {
  sessionId: string;
  message: OutgoingMessage;
  sendMessage: SendMessageFn;
  /** Awaited once the uploads are done and before the send: `true` sends,
   *  `false` drops the message. The placeholder shows the whole time. */
  waitUntilReady?: () => Promise<boolean>;
}

export function hasAttachments(message: OutgoingMessage): boolean {
  return message.files.length > 0 || message.parts.length > 0;
}

/** Backstop for the composer's own caps; toasts and returns false when the
 *  message cannot go out. */
export function validateAttachments(message: OutgoingMessage): boolean {
  // Workspace references count too — uploaded or not, every attachment
  // becomes one `file_ids` entry on the request the backend bounds.
  if (message.files.length + message.parts.length > MAX_ATTACHMENTS) {
    toast({
      title: "Too many attachments",
      description: `You can attach up to ${MAX_ATTACHMENTS} attachments per message.`,
      variant: "destructive",
    });
    return false;
  }
  const oversized = message.files.filter((f) => f.size > MAX_FILE_SIZE_BYTES);
  if (oversized.length > 0) {
    toast({
      title: "File too large",
      description: `${oversized[0].name} exceeds the 100 MB limit.`,
      variant: "destructive",
    });
    return false;
  }
  return true;
}

/**
 * Uploads the message's local files into the session, then sends the message
 * with every attachment as a file part. While that runs, the session's
 * placeholder bubble shows the message with each file's own upload status.
 * A file that fails to upload holds the send until the user retries it or
 * removes it (`retryPendingUpload` / `removePendingUpload`). Clearing the
 * placeholder cancels the send: once it is gone the message never goes out.
 */
export async function sendWithAttachments({
  sessionId,
  message,
  sendMessage,
  waitUntilReady,
}: SendArgs) {
  const { text, files, parts, metadata } = message;
  // The per-click UUID that becomes the backend's ``ChatMessage.id``
  // is generated inside the transport's ``prepareSendMessagesRequest``
  // (one call per ``sendMessages``, stable across SDK-internal
  // retries) — NOT here.  AI SDK's ``messageId`` arg means
  // "replace-existing-message" (edit mode), not "id of a new
  // message", so passing it would put the SDK into edit-mode with no
  // target and break the optimistic-render path that pushes the user
  // bubble into ``messages`` synchronously.
  if (files.length === 0 && !waitUntilReady) {
    await sendMessage(messageInput(text, parts, metadata));
    return;
  }
  // The bubble shows right away; the transcript reads "Uploading N files…"
  // until the uploads land and the real message takes the placeholder's
  // spot. The slot is set here because this is the one path every send with
  // files goes through, including the flush after session creation.
  const pending = describePendingUpload(message);
  useCopilotStreamStore.getState().setPendingUploadSend(sessionId, pending);
  const localAttachments = pending.attachments.slice(parts.length);
  let send: Promise<void> | undefined;
  try {
    const uploads = await Promise.all(
      files.map((file, i) =>
        uploadUntilSettled(sessionId, pending, localAttachments[i].id, file),
      ),
    );
    if (uploads.includes(ABANDONED) || !isPending(sessionId, pending)) return;
    const sentParts = [
      ...parts,
      ...uploads.filter((part): part is FileUIPart => part !== null),
    ];
    if (!text && sentParts.length === 0) return;
    if (waitUntilReady && !(await waitUntilReady())) return;
    if (!isPending(sessionId, pending)) return;
    send = sendMessage(messageInput(text, sentParts, metadata));
  } finally {
    // `sendMessage` pushes the user bubble into `messages` synchronously,
    // so the placeholder can go in the same tick with no gap between the
    // two. Its promise only settles when the whole stream ends, which is
    // why the cleanup happens here and not after the await below.
    useCopilotStreamStore.getState().clearPendingUploadSend(sessionId, pending);
  }
  await send;
}

export function describePendingUpload(
  message: OutgoingMessage,
): PendingUploadSend {
  const stored: PendingUploadAttachment[] = message.parts.map((part) =>
    isFolderPart(part)
      ? {
          id: nextPendingUploadId(),
          name: part.data.name,
          mediaType: "folder",
          status: "uploaded",
        }
      : {
          id: nextPendingUploadId(),
          name: part.filename ?? "file",
          mediaType: part.mediaType,
          status: "uploaded",
        },
  );
  const local: PendingUploadAttachment[] = message.files.map((file) => ({
    id: nextPendingUploadId(),
    name: file.name,
    mediaType: file.type || "application/octet-stream",
    sizeBytes: file.size,
    status: "uploading",
  }));
  return {
    id: nextPendingUploadId(),
    text: message.text,
    attachments: [...stored, ...local],
  };
}

let pendingUploadCount = 0;

function nextPendingUploadId() {
  pendingUploadCount += 1;
  return `pending-upload-${pendingUploadCount}`;
}

type FailedUploadChoice = "retry" | "remove";

const ABANDONED = Symbol("abandoned");

/** What each failed card's Retry and Remove resolve, by attachment id. An
 *  entry lives exactly as long as its card shows the failed state. */
const failedUploads = new Map<
  string,
  (choice: FailedUploadChoice | typeof ABANDONED) => void
>();

export function retryPendingUpload(attachmentId: string) {
  failedUploads.get(attachmentId)?.("retry");
}

export function removePendingUpload(attachmentId: string) {
  failedUploads.get(attachmentId)?.("remove");
}

/** Resolves to the file's part once it is stored, `null` when the user
 *  removed it, or `ABANDONED` when its placeholder went away first. */
async function uploadUntilSettled(
  sessionId: string,
  pending: PendingUploadSend,
  attachmentId: string,
  file: File,
): Promise<FileUIPart | null | typeof ABANDONED> {
  const store = useCopilotStreamStore.getState();
  for (;;) {
    store.setPendingUploadAttachment(sessionId, attachmentId, {
      status: "uploading",
    });
    try {
      const uploaded = await uploadFile(file, sessionId);
      store.setPendingUploadAttachment(sessionId, attachmentId, {
        status: "uploaded",
      });
      return buildFilePart(uploaded);
    } catch (err) {
      console.error("File upload failed:", err);
      store.setPendingUploadAttachment(sessionId, attachmentId, {
        status: "failed",
      });
    }
    const choice = await waitForChoice(sessionId, pending, attachmentId);
    if (choice === "remove") {
      store.removePendingUploadAttachment(sessionId, attachmentId);
      return null;
    }
    if (choice === ABANDONED) return ABANDONED;
  }
}

function waitForChoice(
  sessionId: string,
  pending: PendingUploadSend,
  attachmentId: string,
) {
  if (!isPending(sessionId, pending)) return Promise.resolve(ABANDONED);
  return new Promise<FailedUploadChoice | typeof ABANDONED>((resolve) => {
    const unsubscribe = useCopilotStreamStore.subscribe(() => {
      if (!isPending(sessionId, pending)) settle(ABANDONED);
    });
    function settle(choice: FailedUploadChoice | typeof ABANDONED) {
      failedUploads.delete(attachmentId);
      unsubscribe();
      resolve(choice);
    }
    failedUploads.set(attachmentId, settle);
  });
}

function isPending(sessionId: string, pending: PendingUploadSend) {
  return (
    useCopilotStreamStore.getState().pendingUploadSends[sessionId]?.id ===
    pending.id
  );
}

async function uploadFile(file: File, sessionId: string) {
  const data = await uploadFileDirect(file, sessionId);
  if (!data.file_id) throw new Error("No file_id returned");
  return {
    file_id: data.file_id,
    name: data.name || file.name,
    mime_type: data.mime_type || "application/octet-stream",
  } satisfies UploadedFile;
}

function buildFilePart(uploaded: UploadedFile): FileUIPart {
  return {
    type: "file",
    mediaType: uploaded.mime_type,
    filename: uploaded.name,
    url: workspaceFileDownloadUrl(uploaded.file_id),
  };
}

function messageInput(
  text: string,
  parts: StoredAttachmentPart[],
  metadata?: ExpertKickoffMetadata,
) {
  // `{ text, files }` only accepts FileUIParts, so a message carrying a
  // folder is built from its parts instead.
  if (!parts.some(isFolderPart)) {
    return {
      text,
      files: parts.length > 0 ? (parts as FileUIPart[]) : undefined,
      metadata,
    };
  }
  return {
    parts: [
      ...parts,
      ...(text ? [{ type: "text" as const, text }] : []),
    ] as UIMessage["parts"],
    metadata,
  };
}
