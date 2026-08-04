"use client";

import {
  CheckCircleIcon,
  PuzzlePieceIcon,
  XCircleIcon,
} from "@phosphor-icons/react";
import Image from "next/image";
import { useState } from "react";
import { CARD, HALF } from "./ResultCards";
import { inline, str } from "./resultHelpers";

export function CardProviderIcon({
  provider,
  fallback,
  size = 15,
}: {
  provider: string | null;
  fallback: React.ReactNode;
  size?: number;
}) {
  const [failed, setFailed] = useState(false);
  if (!provider || failed) return <>{fallback}</>;
  const slug = provider
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, "_");
  return (
    <Image
      src={`/integrations/${slug}.png`}
      alt={provider}
      width={size}
      height={size}
      className="rounded-sm object-contain"
      onError={() => setFailed(true)}
    />
  );
}

const BLOCK_ICON = <PuzzlePieceIcon size={15} className="text-zinc-600" />;

export function BlockListCard({
  blocks,
}: {
  blocks: Record<string, unknown>[];
}) {
  return (
    <div className="grid gap-1.5 sm:grid-cols-2">
      {blocks.map((block, i) => {
        const categories = Array.isArray(block.categories)
          ? (block.categories as unknown[]).filter(
              (c): c is string => typeof c === "string",
            )
          : [];
        return (
          <div key={i} className={CARD + " flex items-start gap-2.5 p-2.5"}>
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

export function BlockOutputCard({
  output,
}: {
  output: Record<string, unknown>;
}) {
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
          <CheckCircleIcon
            size={16}
            weight="fill"
            className="shrink-0 text-green-500"
          />
        ) : (
          <XCircleIcon
            size={16}
            weight="fill"
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
