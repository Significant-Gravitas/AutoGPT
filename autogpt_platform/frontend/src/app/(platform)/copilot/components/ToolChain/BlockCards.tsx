"use client";

import {
  CancelCircleIcon,
  CheckmarkCircle02Icon,
  PuzzleIcon,
} from "@hugeicons/core-free-icons";
import Image from "next/image";
import { useState } from "react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { CARD, HALF } from "./ResultCards";
import {
  inline,
  integrationIconSrc,
  resultItemKey,
  str,
} from "./resultHelpers";

interface CardProviderIconProps {
  provider: string | null;
  fallback: React.ReactNode;
  size?: number;
}

interface BlockListCardProps {
  blocks: Record<string, unknown>[];
}

interface BlockOutputCardProps {
  output: Record<string, unknown>;
}

export function CardProviderIcon({
  provider,
  fallback,
  size = 15,
}: CardProviderIconProps) {
  const [failed, setFailed] = useState(false);
  if (!provider || failed) return <>{fallback}</>;
  const src = integrationIconSrc(provider);
  if (!src) return <>{fallback}</>;
  return (
    <Image
      src={src}
      alt={provider}
      width={size}
      height={size}
      className="rounded-sm object-contain"
      onError={() => setFailed(true)}
    />
  );
}

const BLOCK_ICON = (
  <Icon icon={PuzzleIcon} size={15} className="text-zinc-600" />
);

export function BlockListCard({ blocks }: BlockListCardProps) {
  return (
    <div className="grid gap-1.5 sm:grid-cols-2">
      {blocks.map((block, i) => {
        const categories = Array.isArray(block.categories)
          ? (block.categories as unknown[]).filter(
              (c): c is string => typeof c === "string",
            )
          : [];
        return (
          <div
            key={resultItemKey(block, i)}
            className={CARD + " flex items-start gap-2.5 p-2.5"}
          >
            <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-zinc-100">
              <CardProviderIcon
                provider={str(block, "provider")}
                fallback={BLOCK_ICON}
              />
            </div>
            <div className="min-w-0 flex-1">
              <p className="truncate text-[13px] font-medium text-zinc-800">
                {str(block, "name", "block_name") ?? inline(block)}
              </p>
              {str(block, "description") && (
                <p className="truncate text-xs text-zinc-500">
                  {str(block, "description")}
                </p>
              )}
            </div>
            {categories[0] && (
              <span className="shrink-0 rounded-full bg-zinc-100 px-2 py-0.5 text-[11px] text-zinc-500">
                {categories[0].toLowerCase()}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}

export function BlockOutputCard({ output }: BlockOutputCardProps) {
  const name = str(output, "block_name", "block_id") ?? "Block";
  const ok = output.success !== false;
  const entries = Object.entries(
    output.outputs && typeof output.outputs === "object"
      ? (output.outputs as Record<string, unknown>)
      : {},
  );
  return (
    <div className={`${CARD} ${HALF} overflow-hidden`}>
      <div className="flex items-center gap-2.5 p-2.5">
        <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-zinc-100">
          <CardProviderIcon
            provider={str(output, "provider")}
            fallback={BLOCK_ICON}
          />
        </div>
        <p className="min-w-0 flex-1 truncate text-[13px] font-medium text-zinc-800">
          {name}
        </p>
        {ok ? (
          <Icon
            icon={CheckmarkCircle02Icon}
            size={16}
            className="shrink-0 text-green-500"
          />
        ) : (
          <Icon
            icon={CancelCircleIcon}
            size={16}
            className="shrink-0 text-red-400"
          />
        )}
      </div>
      {entries.length > 0 && (
        <div className="divide-y divide-zinc-100 border-t border-zinc-100">
          {entries.map(([key, value]) => {
            const flat =
              Array.isArray(value) && value.length === 1 ? value[0] : value;
            return (
              <div key={key} className="px-2.5 py-1.5">
                <p className="text-[11px] uppercase tracking-wide text-zinc-400">
                  {key.replace(/_/g, " ")}
                </p>
                <p className="mt-0.5 line-clamp-3 whitespace-pre-wrap break-words text-[13px] text-zinc-700">
                  {inline(flat)}
                </p>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
