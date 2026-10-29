import { isPaywallError } from "@/components/contextual/PaywallToast/helpers";
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
 * placeholder bubble shows the message and reads "Uploading N files…".
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
  let send: Promise<void> | undefined;
  try {
    const uploaded = await uploadFiles(files, sessionId);
    let sentParts = [...parts, ...buildFileParts(uploaded)];
    if (files.length > 0 && uploaded.length === 0) {
      toast({
        title: "File upload failed",
        description: "Could not upload any files. Please try again.",
        variant: "destructive",
      });
      // The workspace references didn't fail to upload (they need no upload),
      // so don't discard them just because the local uploads failed.
      if (parts.length === 0) {
        throw new Error("All file uploads failed");
      }
      sentParts = parts;
    }
    if (waitUntilReady && !(await waitUntilReady())) return;
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
      ? { name: part.data.name, mediaType: "folder", isUploading: false }
      : {
          name: part.filename ?? "file",
          mediaType: part.mediaType,
          isUploading: false,
          url: part.url,
        },
  );
  const local: PendingUploadAttachment[] = message.files.map((file) => ({
    name: file.name,
    mediaType: file.type || "application/octet-stream",
    sizeBytes: file.size,
    isUploading: true,
    file,
  }));
  return { text: message.text, attachments: [...stored, ...local] };
}

async function uploadFiles(
  files: File[],
  sessionId: string,
): Promise<UploadedFile[]> {
  const results = await Promise.allSettled(
    files.map(async (file) => {
      try {
        const data = await uploadFileDirect(file, sessionId);
        if (!data.file_id) throw new Error("No file_id returned");
        return {
          file_id: data.file_id,
          name: data.name || file.name,
          mime_type: data.mime_type || file.type || "application/octet-stream",
        } as UploadedFile;
      } catch (err) {
        // The send's own failure toast says a plan is needed.
        if (isPaywallError(err)) throw err;
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
  const paywalled = results.find(
    (r): r is PromiseRejectedResult =>
      r.status === "rejected" && isPaywallError(r.reason),
  );
  if (paywalled) throw paywalled.reason;
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
