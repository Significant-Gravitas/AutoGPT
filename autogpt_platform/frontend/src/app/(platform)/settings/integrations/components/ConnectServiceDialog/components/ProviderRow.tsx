"use client";

import { useState } from "react";
import { PlusIcon } from "@phosphor-icons/react";

import type { ConnectableProvider } from "../helpers";

interface Props {
  provider: ConnectableProvider;
  onSelect: (id: string) => void;
}

export function ProviderRow({ provider, onSelect }: Props) {
  const [broken, setBroken] = useState(false);
  const src = `/integrations/${provider.id}.png`;

  return (
    <button
      type="button"
      onClick={() => onSelect(provider.id)}
      className="group flex h-16 w-full items-center gap-3 rounded-xl border border-zinc-200 bg-white px-[0.875rem] py-[0.625rem] text-left transition-colors hover:bg-zinc-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-400 active:bg-zinc-100 active:ring-1 active:ring-zinc-300"
    >
      {broken ? (
        <div aria-hidden className="size-9 shrink-0 rounded-md bg-zinc-100" />
      ) : (
        /* eslint-disable-next-line @next/next/no-img-element -- decorative provider logo, no LCP candidate */
        <img
          src={src}
          alt=""
          width={36}
          height={36}
          loading="lazy"
          decoding="async"
          className="size-9 shrink-0 object-contain"
          onError={() => setBroken(true)}
        />
      )}
      <span className="flex min-w-0 flex-1 flex-col gap-0.5">
        <span className="truncate text-[14px] font-medium leading-[22px] text-zinc-800">
          {provider.name}
        </span>
        <span className="truncate text-[12px] leading-[20px] text-zinc-500">
          {provider.description ?? provider.id}
        </span>
      </span>
      <span
        aria-hidden
        className="flex size-7 shrink-0 items-center justify-center rounded-lg bg-zinc-700 text-white transition-transform group-hover:bg-zinc-800 group-active:scale-[0.96]"
      >
        <PlusIcon size={18} weight="bold" />
      </span>
    </button>
  );
}
