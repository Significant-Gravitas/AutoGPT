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
      className="inline-flex size-3.5 shrink-0 items-center justify-center"
      title={`From ${logo.name}`}
    >
      <Image
        src={logo.src}
        alt={logo.name}
        width={14}
        height={14}
        loading="lazy"
        className="size-3.5 object-contain opacity-80 grayscale"
        onError={() => setBrokenSrc(logo.src)}
      />
    </span>
  );
}
