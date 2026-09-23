"use client";

import { isRenderableImageUrl } from "@/lib/next-image";
import Image from "next/image";
import { isLocalStoreMediaUrl } from "@/lib/store-media";
import { useState } from "react";

interface Props {
  imageUrl?: string | null;
}

/** The workflow's image, or a plain muted square when there is none, when
 *  next/image cannot load it, or when it fails to load: never a broken-image
 *  outline, never a stand-in glyph. */
export function WorkflowTile({ imageUrl }: Props) {
  const [hasError, setHasError] = useState(false);
  if (!isRenderableImageUrl(imageUrl) || hasError) {
    return (
      <span
        aria-hidden="true"
        className="size-9 shrink-0 rounded-lg bg-zinc-100"
      />
    );
  }
  return (
    <Image
      src={imageUrl}
      unoptimized={isLocalStoreMediaUrl(imageUrl)}
      alt=""
      width={36}
      height={36}
      onError={() => setHasError(true)}
      className="size-9 shrink-0 rounded-lg bg-zinc-100 object-cover"
    />
  );
}
