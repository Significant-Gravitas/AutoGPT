"use client";

import { useState } from "react";

interface Props {
  id: string;
  name: string;
}

export function ProviderAvatar({ id, name }: Props) {
  const [broken, setBroken] = useState(false);
  if (broken) {
    return (
      <div aria-hidden className="size-10 shrink-0 rounded-md bg-zinc-100" />
    );
  }
  return (
    /* eslint-disable-next-line @next/next/no-img-element -- decorative provider logo, no LCP candidate */
    <img
      src={`/integrations/${id}.png`}
      alt={`${name} logo`}
      width={40}
      height={40}
      loading="lazy"
      decoding="async"
      onError={() => setBroken(true)}
      className="size-10 shrink-0 object-contain"
    />
  );
}
