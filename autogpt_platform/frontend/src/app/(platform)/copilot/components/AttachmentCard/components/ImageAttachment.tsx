import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import {
  Cancel01Icon,
  Image01Icon,
  Loading03Icon,
} from "@hugeicons/core-free-icons";
import type { AttachmentCardProps } from "../AttachmentCard";
import { CardActionButton } from "./CardActionButton";
import { CardPressable, isPressable } from "./CardPressable";
import { UploadStatusText } from "./UploadStatusText";

interface Props extends AttachmentCardProps {
  onPreviewError: () => void;
}

export function ImageAttachment({
  name,
  previewUrl,
  status,
  size,
  onRemove,
  onOpen,
  href,
  onPreviewError,
}: Props) {
  return (
    <div
      className={cn(
        "relative rounded-2xl bg-zinc-100",
        size === "composer" ? "size-14" : "size-36",
      )}
    >
      <CardPressable
        onOpen={onOpen}
        href={href}
        title={name}
        className={cn(
          "flex size-full items-center justify-center overflow-hidden rounded-2xl focus-visible:ring-offset-2",
          isPressable({ onOpen, href }) &&
            "transition-opacity hover:opacity-90",
        )}
      >
        {previewUrl ? (
          // eslint-disable-next-line @next/next/no-img-element -- blob: and authed proxy URLs
          <img
            src={previewUrl}
            alt={name}
            onError={onPreviewError}
            loading="lazy"
            decoding="async"
            className="size-full object-cover"
          />
        ) : (
          <>
            <Icon
              icon={Image01Icon}
              size={20}
              aria-hidden
              className="text-zinc-400"
            />
            <span className="sr-only">{name}</span>
          </>
        )}
      </CardPressable>
      <span
        aria-hidden
        className="pointer-events-none absolute inset-0 rounded-2xl outline outline-1 -outline-offset-1 outline-black/10"
      />
      {status === "uploading" && (
        <span className="pointer-events-none absolute inset-0 flex items-center justify-center rounded-2xl bg-black/40">
          <Icon
            icon={Loading03Icon}
            size={20}
            aria-hidden
            className="animate-spin text-white motion-reduce:animate-none"
          />
        </span>
      )}
      <UploadStatusText status={status} />
      {onRemove && (
        <CardActionButton
          size="icon-xs"
          label={`Remove ${name}`}
          onClick={onRemove}
          className="absolute right-1 top-1 size-6 rounded-full border-transparent bg-zinc-900/70 text-white hover:border-transparent hover:bg-zinc-900"
        >
          <Icon icon={Cancel01Icon} size={12} aria-hidden />
        </CardActionButton>
      )}
    </div>
  );
}
