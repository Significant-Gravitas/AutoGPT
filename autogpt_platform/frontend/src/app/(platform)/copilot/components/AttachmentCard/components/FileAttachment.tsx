import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import {
  AlertCircleIcon,
  Cancel01Icon,
  Loading03Icon,
  RefreshIcon,
} from "@hugeicons/core-free-icons";
import type { AttachmentCardProps } from "../AttachmentCard";
import { attachmentDetail, attachmentIcon } from "../helpers";
import { CardActionButton } from "./CardActionButton";
import { CardPressable, isPressable } from "./CardPressable";
import { UploadStatusText } from "./UploadStatusText";

export function FileAttachment({
  name,
  mediaType,
  sizeBytes,
  detail,
  status,
  size,
  onRemove,
  onRetry,
  onOpen,
  href,
}: AttachmentCardProps) {
  const isFailed = status === "failed";
  const isUploading = status === "uploading";
  const icon = isFailed
    ? AlertCircleIcon
    : isUploading
      ? Loading03Icon
      : attachmentIcon(mediaType, name);

  return (
    <div
      className={cn(
        "flex max-w-full items-center gap-1 rounded-2xl border bg-white p-2",
        size === "composer" ? "w-60" : "w-72",
        isFailed ? "border-red-200" : "border-zinc-200",
      )}
    >
      <CardPressable
        onOpen={onOpen}
        href={href}
        title={name}
        className={cn(
          "flex min-w-0 flex-1 items-center gap-3 rounded-lg text-left",
          isPressable({ onOpen, href }) && "transition-colors hover:bg-zinc-50",
        )}
      >
        <span
          className={cn(
            "flex size-10 shrink-0 items-center justify-center rounded-lg",
            isFailed ? "bg-red-50 text-red-600" : "bg-zinc-100 text-zinc-600",
          )}
        >
          <Icon
            icon={icon}
            size={20}
            aria-hidden
            className={cn(
              isUploading && "animate-spin motion-reduce:animate-none",
            )}
          />
        </span>
        <span className="flex min-w-0 flex-1 flex-col">
          <Text
            variant="body-medium"
            as="span"
            tone="primary"
            unmask={false}
            className="truncate"
          >
            {name}
          </Text>
          <Text
            variant={isFailed ? "small-medium" : "small"}
            as="span"
            tone={isFailed ? "danger" : "secondary"}
            unmask={isFailed}
            className="truncate"
          >
            {isFailed
              ? "Upload failed"
              : (detail ?? attachmentDetail(mediaType, name, sizeBytes))}
          </Text>
        </span>
        <UploadStatusText status={status} />
      </CardPressable>
      {isFailed && onRetry && (
        <CardActionButton
          size="xs"
          leadingIcon={RefreshIcon}
          label={`Retry ${name}`}
          onClick={onRetry}
        >
          <span className="hidden sm:inline">Retry</span>
        </CardActionButton>
      )}
      {onRemove && (
        <CardActionButton
          size="icon-xs"
          label={`Remove ${name}`}
          onClick={onRemove}
          className="text-zinc-500 hover:bg-zinc-100 hover:text-zinc-900"
        >
          <Icon icon={Cancel01Icon} size={14} aria-hidden />
        </CardActionButton>
      )}
    </div>
  );
}
