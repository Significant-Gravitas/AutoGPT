"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { Add01Icon, CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";
import Image from "next/image";
import { useState } from "react";

import type { ConnectableProvider } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/helpers";

interface Props {
  provider: ConnectableProvider;
  onSelect: (id: string) => void;
  isConnected?: boolean;
  /** Second line, used by the recommended list to show the model's reason. */
  description?: string | null;
  className?: string;
}

// The welcome dialog's provider card: compact (two per row), name plus an
// optional one-line reason, with a Connected state once credentials exist
// for the provider.
export function ConnectProviderRow({
  provider,
  onSelect,
  isConnected,
  description,
  className,
}: Props) {
  const src = `/integrations/${provider.id}.png`;
  const [brokenSrc, setBrokenSrc] = useState<string | null>(null);
  const broken = brokenSrc === src;

  return (
    <button
      type="button"
      onClick={() => onSelect(provider.id)}
      className={cn(
        "group flex h-14 w-full items-center gap-2.5 rounded-md border border-zinc-200 bg-white px-3 text-left transition-colors hover:bg-zinc-50 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-zinc-900 active:bg-zinc-100",
        className,
      )}
    >
      {broken ? (
        <div
          aria-hidden
          className="flex size-7 shrink-0 items-center justify-center rounded-md bg-zinc-100 text-[12px] font-semibold uppercase text-zinc-600"
        >
          {provider.name?.charAt(0) ?? provider.id.charAt(0)}
        </div>
      ) : (
        <Image
          src={src}
          alt=""
          width={28}
          height={28}
          loading="lazy"
          className="size-7 shrink-0 object-contain"
          onError={() => setBrokenSrc(src)}
        />
      )}
      <span className="flex min-w-0 flex-1 flex-col">
        <span className="flex min-w-0 items-center gap-1.5">
          <span className="truncate text-[14px] font-medium leading-[22px] text-zinc-800">
            {provider.name}
          </span>
          {/* Connected mark beside the title — the + stays because a
              provider can hold multiple credentials. */}
          {isConnected && (
            <Icon
              icon={CheckmarkCircle02Icon}
              size={15}
              className="shrink-0 text-emerald-500"
            />
          )}
        </span>
        {description && (
          <span className="truncate text-[12px] leading-[16px] text-zinc-500">
            {description}
          </span>
        )}
      </span>
      <span
        aria-hidden
        className="flex size-6 shrink-0 items-center justify-center rounded-md bg-zinc-100 text-zinc-500 transition-colors group-hover:bg-zinc-200 group-hover:text-zinc-700"
      >
        <Icon icon={Add01Icon} size={14} />
      </span>
    </button>
  );
}
