"use client";

import { CredentialMentionText } from "../../CredentialMention/CredentialMentionText";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Message, MessageContent } from "@/components/ai-elements/message";
import { cn } from "@/lib/utils";
import { Loading03Icon } from "@hugeicons/core-free-icons";
import type {
  PendingUploadAttachment,
  PendingUploadSend,
} from "../../../copilotStreamStore";
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
  const count = attachments.filter((a) => a.isUploading).length;
  if (count === 0) return "Sending…";
  return `Uploading ${count} ${count === 1 ? "file" : "files"}…`;
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
  return (
    <div className="my-1 flex w-full min-w-0 items-center gap-3 rounded-2xl border border-zinc-200 bg-white px-3 py-2.5 text-left">
      <Icon
        icon={classification.icon}
        size={20}
        className="shrink-0 text-zinc-400"
      />
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-zinc-900">
          {attachment.name}
        </p>
        <p className="text-xs text-zinc-400">
          <span className="inline-block rounded-full bg-blue-50 px-1.5 py-0.5 text-xs font-medium text-blue-500">
            {classification.label}
          </span>
          {attachment.sizeBytes ? ` • ${formatSize(attachment.sizeBytes)}` : ""}
        </p>
      </div>
      {attachment.isUploading && (
        <Icon
          icon={Loading03Icon}
          size={16}
          className="shrink-0 animate-spin text-zinc-400"
        />
      )}
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
              "group-[.is-user]:bg-zinc-100 group-[.is-user]:px-4 group-[.is-user]:py-2.5 group-[.is-user]:text-zinc-900",
            )}
          >
            <div className="whitespace-pre-wrap break-words">
              <CredentialMentionText text={pendingSend.text} />
            </div>
          </MessageContent>
        )}
        {pendingSend.attachments.length > 0 && (
          <div className="mt-2 flex flex-col gap-2">
            {pendingSend.attachments.map((attachment, i) => (
              <PendingArtifactCard
                key={`${attachment.name}-${i}`}
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
          <ThinkingIndicator
            active
            elapsedSeconds={elapsedSeconds}
            statusMessage={label}
          />
        </MessageContent>
      </Message>
    </>
  );
}
