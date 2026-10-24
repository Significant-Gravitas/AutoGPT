"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { Loading03Icon } from "@hugeicons/core-free-icons";
import { type ReactNode, useState } from "react";
import type { PendingUploadAttachment } from "../../../copilotStreamStore";
import type { ArtifactRef } from "../../../store";
import { useCopilotUIStore } from "../../../store";
import { ArtifactCard } from "../../ArtifactCard/ArtifactCard";
import { LocalFileImage } from "../../LocalFileImage/LocalFileImage";

// A fixed size, not the image's own width, so the frame doesn't move while
// the sent image downloads or when it replaces the pending one.
const FRAME = "overflow-hidden rounded-2xl border border-zinc-200 bg-zinc-100";
const IMAGE = "block size-40 object-cover";

interface Props {
  artifact: ArtifactRef;
  readOnly?: boolean;
}

/** An image the user sent, shown as a picture that opens in the side panel. */
export function MessageImageThumbnail({ artifact, readOnly }: Props) {
  const openArtifact = useCopilotUIStore((s) => s.openArtifact);
  const [failed, setFailed] = useState(false);

  if (failed) return <ArtifactCard artifact={artifact} readOnly={readOnly} />;

  return (
    <button
      type="button"
      onClick={() => openArtifact(artifact)}
      aria-label={`Open ${artifact.title}`}
      className={cn(
        FRAME,
        "transition-opacity hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400",
      )}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={artifact.sourceUrl}
        alt={artifact.title}
        className={IMAGE}
        loading="lazy"
        decoding="async"
        onError={() => setFailed(true)}
      />
    </button>
  );
}

interface PendingProps {
  attachment: PendingUploadAttachment;
  /** Shown instead when the image fails to load. */
  fallback: ReactNode;
}

/** The same picture while the message's files upload, so nothing moves when
 *  the sent message replaces the placeholder. */
export function PendingImageThumbnail({ attachment, fallback }: PendingProps) {
  const [failed, setFailed] = useState(false);

  if (failed) return fallback;

  return (
    <div className={cn(FRAME, "relative")}>
      {attachment.file ? (
        <LocalFileImage
          file={attachment.file}
          alt={attachment.name}
          className={IMAGE}
          onError={() => setFailed(true)}
        />
      ) : (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={attachment.url}
          alt={attachment.name}
          className={IMAGE}
          onError={() => setFailed(true)}
        />
      )}
      {attachment.isUploading && (
        <span className="absolute inset-0 flex items-center justify-center bg-black/20">
          <Icon
            icon={Loading03Icon}
            size={20}
            className="animate-spin text-white"
          />
        </span>
      )}
    </div>
  );
}
