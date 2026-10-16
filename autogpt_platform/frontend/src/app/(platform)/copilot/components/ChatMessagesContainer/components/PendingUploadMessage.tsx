"use client";

import { CredentialMentionText } from "../../CredentialMention/CredentialMentionText";
import { Message, MessageContent } from "@/components/ai-elements/message";
import { cn } from "@/lib/utils";
import type {
  PendingUploadAttachment,
  PendingUploadSend,
  PendingUploadStatus,
} from "../../../copilotStreamStore";
import {
  removePendingUpload,
  retryPendingUpload,
} from "../../../helpers/sendWithAttachments";
import { AttachmentCard } from "../../AttachmentCard/AttachmentCard";
import { isImageType } from "../../AttachmentCard/helpers";
import { useObjectUrl } from "../../AttachmentCard/useObjectUrl";
import { useElapsedTimer } from "../../JobStatsBar/useElapsedTimer";
import { ThinkingIndicator } from "./ThinkingIndicator";
import { UserAttachmentRow } from "./UserAttachmentRow";

interface Props {
  pendingSend: PendingUploadSend;
  isCompact?: boolean;
}

/** Stable id so the tail spacer measures the placeholder like a real user
 *  message; the bubble then keeps its spot when the persisted one lands. */
export const PENDING_UPLOAD_MESSAGE_ID = "pending-upload-message";

export function uploadStatusLabel(attachments: PendingUploadAttachment[]) {
  const uploading = countWithStatus(attachments, "uploading");
  if (uploading > 0)
    return `Uploading ${uploading} ${uploading === 1 ? "file" : "files"}…`;
  const failed = attachments.filter((a) => a.status === "failed");
  if (failed.length === 1)
    return `Couldn't upload ${failed[0].name}. Retry or remove it to send your message.`;
  if (failed.length > 1)
    return `Couldn't upload ${failed.length} files. Retry or remove them to send your message.`;
  return "Sending…";
}

function countWithStatus(
  attachments: PendingUploadAttachment[],
  status: PendingUploadStatus,
) {
  return attachments.filter((a) => a.status === status).length;
}

interface CardProps {
  attachment: PendingUploadAttachment;
}

function PendingAttachmentCard({ attachment }: CardProps) {
  const localImage =
    attachment.file && isImageType(attachment.mediaType)
      ? attachment.file
      : undefined;
  const localPreviewUrl = useObjectUrl(localImage);
  const isFailed = attachment.status === "failed";
  return (
    <AttachmentCard
      size="message"
      name={attachment.name}
      mediaType={attachment.mediaType}
      sizeBytes={attachment.sizeBytes}
      previewUrl={localPreviewUrl ?? attachment.previewUrl}
      status={attachment.status}
      onRetry={isFailed ? () => retryPendingUpload(attachment.id) : undefined}
      onRemove={isFailed ? () => removePendingUpload(attachment.id) : undefined}
    />
  );
}

/**
 * Optimistic stand-in for a user message whose local attachments are still
 * uploading. Mirrors the real bubble and attachment cards so the swap to the
 * persisted message is invisible, and narrates the upload in the same spot
 * the backend's own status lines ("Preparing workspace…") appear next.
 */
export function PendingUploadMessage({ pendingSend, isCompact }: Props) {
  const { elapsedSeconds } = useElapsedTimer(true);
  const label = uploadStatusLabel(pendingSend.attachments);
  const isWaitingOnUser =
    countWithStatus(pendingSend.attachments, "uploading") === 0 &&
    countWithStatus(pendingSend.attachments, "failed") > 0;

  return (
    <>
      <Message
        from="user"
        data-testid="pending-upload-message"
        data-message-id={PENDING_UPLOAD_MESSAGE_ID}
        className="duration-300 animate-in fade-in slide-in-from-bottom-2 fill-mode-both"
      >
        {pendingSend.attachments.length > 0 && (
          <UserAttachmentRow>
            {pendingSend.attachments.map((attachment) => (
              <PendingAttachmentCard
                key={attachment.id}
                attachment={attachment}
              />
            ))}
          </UserAttachmentRow>
        )}
        {pendingSend.text && (
          <MessageContent
            className={cn(
              isCompact
                ? "text-sm leading-6 group-[.is-user]:rounded-xl"
                : "text-[1rem] leading-relaxed group-[.is-user]:rounded-3xl",
              "group-[.is-user]:bg-zinc-900 group-[.is-user]:px-4 group-[.is-user]:py-2.5 group-[.is-user]:text-zinc-50",
            )}
          >
            <div className="whitespace-pre-wrap break-words">
              <CredentialMentionText text={pendingSend.text} />
            </div>
          </MessageContent>
        )}
      </Message>
      <Message
        from="assistant"
        className="duration-300 animate-in fade-in slide-in-from-bottom-2 fill-mode-both"
      >
        <MessageContent className="text-[1rem] leading-relaxed">
          {/* Announce the status on its own: the indicator's elapsed timer
              ticks every second, so a live region around the whole thing
              would re-read the upload state on every tick. */}
          <span className="sr-only" role="status" aria-live="polite">
            {label}
          </span>
          {isWaitingOnUser ? (
            <p aria-hidden className="text-sm text-zinc-600">
              {label}
            </p>
          ) : (
            <ThinkingIndicator
              active
              elapsedSeconds={elapsedSeconds}
              statusMessage={label}
            />
          )}
        </MessageContent>
      </Message>
    </>
  );
}
