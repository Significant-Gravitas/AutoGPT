"use client";

import Image from "next/image";
import { useState } from "react";
import { PuzzleIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { integrationIconSrc } from "./helpers";

interface Props {
  provider: string;
  /** Accessible label; defaults to the provider slug. */
  alt?: string;
  size?: number;
  className?: string;
}

/** A provider's logo, degrading to a neutral glyph.
 *
 * Providers are added faster than their PNGs are, so a missing file is normal
 * rather than exceptional — it falls back instead of rendering a broken image.
 */
export function IntegrationLogo({
  provider,
  alt,
  size = 16,
  className,
}: Props) {
  const [failed, setFailed] = useState(false);
  const src = integrationIconSrc(provider);

  // The fallback keeps the caller's label: a provider whose PNG is missing is
  // still a named integration, and dropping the label would make it invisible
  // to anyone reading the page with a screen reader.
  if (!src || failed) {
    return (
      <Icon
        icon={PuzzleIcon}
        size={size}
        role="img"
        aria-label={alt ?? provider}
        className={cn("text-zinc-400", className)}
      />
    );
  }

  return (
    <Image
      src={src}
      alt={alt ?? provider}
      width={size}
      height={size}
      className={cn("rounded-sm object-contain", className)}
      onError={() => setFailed(true)}
    />
  );
}
