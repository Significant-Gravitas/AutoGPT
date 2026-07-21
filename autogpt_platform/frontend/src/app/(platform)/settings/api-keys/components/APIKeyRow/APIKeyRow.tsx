"use client";

import { useState } from "react";
import {
  CheckSquareIcon,
  InfoIcon,
  SquareIcon,
  TrashIcon,
} from "@phosphor-icons/react";

import type { APIKeyInfo } from "@/app/api/__generated__/models/aPIKeyInfo";
import { Text } from "@/components/atoms/Text/Text";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";

import { APIKeyInfoDialog } from "../APIKeyInfoDialog/APIKeyInfoDialog";
import { formatLastUsed, maskAPIKey } from "../APIKeyList/helpers";

interface Props {
  apiKey: APIKeyInfo;
  selected: boolean;
  onToggleSelected: () => void;
  onDelete: () => void;
}

export function APIKeyRow({
  apiKey,
  selected,
  onToggleSelected,
  onDelete,
}: Props) {
  const [infoOpen, setInfoOpen] = useState(false);
  const maskedKey = maskAPIKey(apiKey.head, apiKey.tail);
  const lastUsedLabel = formatLastUsed(apiKey.last_used_at);

  return (
    <div
      data-selected={selected}
      className="flex items-center justify-between py-4 pl-3 pr-5 transition-colors data-[selected=true]:bg-zinc-100"
    >
      <div className="flex items-center gap-3">
        <button
          type="button"
          role="checkbox"
          aria-checked={selected}
          aria-label={`Select ${apiKey.name}`}
          onClick={onToggleSelected}
          className={`shrink-0 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-zinc-800 ${
            selected
              ? "text-zinc-800 hover:text-zinc-900"
              : "text-zinc-500 hover:text-zinc-700"
          }`}
        >
          {selected ? (
            <CheckSquareIcon size={20} weight="fill" />
          ) : (
            <SquareIcon size={20} />
          )}
        </button>
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2">
            <Text variant="body-medium" as="span" className="text-textBlack">
              {apiKey.name}
            </Text>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    aria-label={`View details for ${apiKey.name}`}
                    onClick={() => setInfoOpen(true)}
                    className="shrink-0 rounded text-zinc-500 transition-colors hover:text-zinc-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-zinc-800"
                  >
                    <InfoIcon size={16} />
                  </button>
                </TooltipTrigger>
                <TooltipContent side="top">View key details</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          <div className="flex items-center gap-3 whitespace-nowrap">
            <Text variant="label" as="span" className="text-zinc-700">
              {maskedKey}
            </Text>
            <Text
              variant="small"
              as="span"
              className="leading-[20px] text-zinc-500"
            >
              {lastUsedLabel}
            </Text>
          </div>
        </div>
      </div>

      <button
        type="button"
        aria-label={`Delete ${apiKey.name}`}
        onClick={onDelete}
        className="shrink-0 rounded text-zinc-500 transition-colors hover:text-zinc-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-zinc-800"
      >
        <TrashIcon size={20} />
      </button>

      <APIKeyInfoDialog
        open={infoOpen}
        apiKey={apiKey}
        onOpenChange={setInfoOpen}
      />
    </div>
  );
}
