"use client";

import { CredentialMentionText } from "../../CredentialMention/CredentialMentionText";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Message, MessageContent } from "@/components/ai-elements/message";
import { cn } from "@/lib/utils";
import { Button } from "@/components/atoms/Button/Button";
import {
  AlertCircleIcon,
  Cancel01Icon,
  CheckmarkCircle02Icon,
  Loading03Icon,
  RefreshIcon,
} from "@hugeicons/core-free-icons";
import type {
  PendingUploadAttachment,
  PendingUploadSend,
  PendingUploadStatus,
} from "../../../copilotStreamStore";
import {
  removePendingUpload,
  retryPendingUpload,
} from "../../../helpers/sendWithAttachments";
import { classifyArtifact } from "../../ArtifactPanel/helpers";
import { useElapsedTimer } from "../../JobStatsBar/useElapsedTimer";
import { ThinkingIndicator } from "./ThinkingIndicator";

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

function formatSize(bytes?: number): string {
  if (!bytes) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

interface CardProps {
  attachment: PendingUploadAttachment;
}

function PendingArtifactCard({ attachment }: CardProps) {
  const classification = classifyArtifact(
    attachment.mediaType,
    attachment.name,
    attachment.sizeBytes,
  );
  const isFailed = attachment.status === "failed";
  return (
    <div
      data-testid="pending-upload-attachment"
      data-status={attachment.status}
      className={cn(
        "my-1 flex w-full min-w-0 items-center gap-3 rounded-2xl border bg-white px-3 py-2.5 text-left",
        isFailed ? "border-red-200" : "border-zinc-200",
      )}
    >
      <Icon
        icon={isFailed ? AlertCircleIcon : classification.icon}
        size={20}
        className={cn("shrink-0", isFailed ? "text-red-600" : "text-zinc-400")}
      />
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-zinc-900">
          {attachment.name}
        </p>
        {isFailed ? (
          <p className="truncate text-xs font-medium text-red-600">
            Upload failed
          </p>
        ) : (
          <p className="text-xs text-zinc-400">
            <span className="inline-block rounded-full bg-blue-50 px-1.5 py-0.5 text-xs font-medium text-blue-500">
              {classification.label}
            </span>
            {attachment.sizeBytes
              ? ` • ${formatSize(attachment.sizeBytes)}`
              : ""}
          </p>
        )}
      </div>
      <UploadStatus attachment={attachment} />
    </div>
  );
}

function UploadStatus({ attachment }: CardProps) {
  if (attachment.status === "uploading")
    return (
      <span className="flex shrink-0 items-center">
        <Icon
          icon={Loading03Icon}
          size={16}
          aria-hidden
          className="animate-spin text-zinc-500 motion-reduce:animate-none"
        />
        <span className="sr-only">Uploading</span>
      </span>
    );
  if (attachment.status === "uploaded")
    return (
      <span className="flex shrink-0 items-center">
        <Icon
          icon={CheckmarkCircle02Icon}
          size={16}
          aria-hidden
          className="text-zinc-500"
        />
        <span className="sr-only">Uploaded</span>
      </span>
    );
  return (
    <div className="flex shrink-0 items-center gap-1">
      <Button
        variant="ghost"
        size="xs"
        leadingIcon={RefreshIcon}
        aria-label={`Retry ${attachment.name}`}
        onClick={() => retryPendingUpload(attachment.id)}
      >
        <span className="hidden sm:inline">Retry</span>
      </Button>
      <Button
        variant="ghost"
        size="icon-xs"
        aria-label={`Remove ${attachment.name}`}
        onClick={() => removePendingUpload(attachment.id)}
      >
        <Icon icon={Cancel01Icon} size={14} aria-hidden />
      </Button>
    </div>
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
        {pendingSend.attachments.length > 0 && (
          <div className="mt-2 flex flex-col gap-2">
            {pendingSend.attachments.map((attachment) => (
              <PendingArtifactCard
                key={attachment.id}
                attachment={attachment}
              />
            ))}
          </div>
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
