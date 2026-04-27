"use client";

import Image from "next/image";
import { useState } from "react";

interface Props {
  id: string;
  name: string;
}

export function ProviderAvatar({ id, name }: Props) {
  const [broken, setBroken] = useState(false);
  if (broken) {
    return (
      <div
        aria-hidden
        className="flex size-10 shrink-0 items-center justify-center rounded-md bg-zinc-100 text-[16px] font-semibold uppercase text-zinc-600"
      >
        {name?.charAt(0) ?? id.charAt(0)}
      </div>
    );
  }
  return (
    <Image
      src={`/integrations/${id}.png`}
      alt={`${name} logo`}
      width={40}
      height={40}
      loading="lazy"
      onError={() => setBroken(true)}
      className="size-10 shrink-0 object-contain"
    />
  );
}
