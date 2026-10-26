"use client";

import {
  CancelCircleIcon,
  CheckmarkCircle02Icon,
  PuzzleIcon,
} from "@hugeicons/core-free-icons";
import Image from "next/image";
import { useState } from "react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { getBlockDisplayName } from "../../helpers/toolDisplay";
import { CARD, HALF, RESULT_GRID } from "./ResultCards";
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
  <Icon icon={PuzzleIcon} size={15} className="text-foreground-secondary" />
);

export function BlockListCard({ blocks }: BlockListCardProps) {
  return (
    <div className={RESULT_GRID}>
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
            <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-surface-hover">
              <CardProviderIcon
                provider={str(block, "provider")}
                fallback={BLOCK_ICON}
              />
            </div>
            <div className="min-w-0 flex-1">
              <p className="truncate text-body-sm font-medium leading-inherit text-foreground-emphasis">
                {getBlockDisplayName(block.name, block) ?? "Block"}
              </p>
              {str(block, "description") && (
                <p className="truncate text-xs text-foreground-tertiary">
                  {str(block, "description")}
                </p>
              )}
            </div>
            {categories[0] && (
              <span className="shrink-0 rounded-full bg-surface-hover px-2 py-0.5 text-caption-sm leading-inherit text-foreground-tertiary">
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
  const name = getBlockDisplayName(undefined, output) ?? "Block";
  const ok = output.success !== false;
  const entries = Object.entries(
    output.outputs && typeof output.outputs === "object"
      ? (output.outputs as Record<string, unknown>)
      : {},
  );
  return (
    <div className={`${CARD} ${HALF} overflow-hidden`}>
      <div className="flex items-center gap-2.5 p-2.5">
        <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-surface-hover">
          <CardProviderIcon
            provider={str(output, "provider")}
            fallback={BLOCK_ICON}
          />
        </div>
        <p className="min-w-0 flex-1 truncate text-body-sm font-medium leading-inherit text-foreground-emphasis">
          {name}
        </p>
        {ok ? (
          <Icon
            icon={CheckmarkCircle02Icon}
            size={16}
            // TODO(tokens): no exact role for text-green-500
            className="shrink-0 text-green-500"
          />
        ) : (
          <Icon
            icon={CancelCircleIcon}
            size={16}
            // TODO(tokens): no exact role for text-red-400
            className="shrink-0 text-red-400"
          />
        )}
      </div>
      {entries.length > 0 && (
        <div className="divide-y divide-separator border-t border-separator">
          {entries.map(([key, value]) => {
            const flat =
              Array.isArray(value) && value.length === 1 ? value[0] : value;
            return (
              <div key={key} className="px-2.5 py-1.5">
                <p className="text-caption-sm uppercase leading-inherit tracking-wide text-foreground-subtle">
                  {key.replace(/_/g, " ")}
                </p>
                <p className="mt-0.5 line-clamp-3 whitespace-pre-wrap break-words text-body-sm leading-inherit text-foreground-body">
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
