"use client";

import Image from "next/image";
import { useState } from "react";
import { cn } from "@/lib/utils";
import { integrationIconSrc } from "@/components/molecules/IntegrationLogo/helpers";
import type { ConnectableProvider } from "../helpers";

const SIZES = {
  sm: { px: 28, box: "size-7", letter: "text-caption" },
  md: { px: 36, box: "size-9", letter: "text-body" },
};

interface Props {
  provider: ConnectableProvider;
  size: keyof typeof SIZES;
}

export function ProviderLogo({ provider, size }: Props) {
  const src = integrationIconSrc(provider.iconId ?? provider.id);
  const [brokenSrc, setBrokenSrc] = useState<string | null>(null);
  const { px, box, letter } = SIZES[size];

  if (!src || brokenSrc === src) {
    return (
      <div
        aria-hidden
        className={cn(
          "flex shrink-0 items-center justify-center rounded-md bg-surface-hover font-semibold uppercase leading-inherit text-foreground-secondary",
          box,
          letter,
        )}
      >
        {provider.name.charAt(0)}
      </div>
    );
  }
  return (
    <Image
      src={src}
      alt=""
      width={px}
      height={px}
      loading="lazy"
      className={cn("shrink-0 object-contain", box)}
      onError={() => setBrokenSrc(src)}
    />
  );
}
