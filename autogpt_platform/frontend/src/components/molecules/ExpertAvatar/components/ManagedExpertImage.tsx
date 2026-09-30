"use client";

import Image from "next/image";
import { useState } from "react";
import { cn } from "@/lib/utils";

interface Props {
  name: string;
  base: string;
  pixels: number;
  /** Largest PNG the library ships; WebP goes to 2x of every size. */
  pngMaxPixels?: number;
  size: number;
  isOtto: boolean;
  className?: string;
}

export function ManagedExpertImage({
  name,
  base,
  pixels,
  pngMaxPixels = 1024,
  size,
  isOtto,
  className,
}: Props) {
  const [format, setFormat] = useState<"webp" | "png" | null>("webp");
  const label = isOtto
    ? "Otto, your personal Head of AI"
    : `${name}, AI Expert`;
  const classes = cn(
    "inline-flex shrink-0 overflow-hidden rounded-xl",
    className,
  );

  if (!format) {
    return (
      <span
        role="img"
        aria-label={label}
        style={{ width: size, height: size }}
        className={cn(
          classes,
          "items-center justify-center bg-muted text-foreground",
        )}
      >
        {name.trim().slice(0, 2).toUpperCase() || "AI"}
      </span>
    );
  }

  return (
    <picture className={classes} style={{ width: size, height: size }}>
      <source
        type={`image/${format}`}
        srcSet={
          format === "png" && pixels * 2 > pngMaxPixels
            ? `${base}/${pixels}.png 1x`
            : `${base}/${pixels}.${format} 1x, ${base}/${pixels * 2}.${format} 2x`
        }
      />
      <Image
        src={`${base}/${pixels}.${format}`}
        alt={label}
        width={size}
        height={size}
        unoptimized
        className="h-full w-full object-contain"
        onError={() => setFormat(format === "webp" ? "png" : null)}
      />
    </picture>
  );
}
