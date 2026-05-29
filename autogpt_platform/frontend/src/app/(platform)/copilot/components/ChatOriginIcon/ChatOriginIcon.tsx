"use client";

import Image from "next/image";
import { useState } from "react";
import { resolvePlatformLogo } from "./platformLogos";

interface Props {
  sourcePlatform?: string | null;
}

export function ChatOriginIcon({ sourcePlatform }: Props) {
  const logo = resolvePlatformLogo(sourcePlatform);
  const [brokenSrc, setBrokenSrc] = useState<string | null>(null);

  if (!logo || brokenSrc === logo.src) return null;

  return (
    <span
      className="inline-flex size-4 shrink-0 items-center justify-center"
      title={`From ${logo.name}`}
    >
      <Image
        src={logo.src}
        alt={logo.name}
        width={16}
        height={16}
        loading="lazy"
        className="size-4 object-contain"
        onError={() => setBrokenSrc(logo.src)}
      />
    </span>
  );
}
