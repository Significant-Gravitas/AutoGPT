"use client";

import { useMountEffect } from "@/hooks/useMountEffect";
import { useState } from "react";

interface Props {
  file: File;
  alt?: string;
  className?: string;
  onError?: () => void;
}

/** Draws a picked file before it uploads. Give it a `key` per file: the blob
 *  URL is made on mount and revoked on unmount. */
export function LocalFileImage({ file, alt = "", className, onError }: Props) {
  const [src, setSrc] = useState<string | null>(null);

  useMountEffect(() => {
    const url = URL.createObjectURL(file);
    setSrc(url);
    return () => URL.revokeObjectURL(url);
  });

  if (!src) return <span aria-hidden className={className} />;
  // eslint-disable-next-line @next/next/no-img-element
  return <img src={src} alt={alt} className={className} onError={onError} />;
}
