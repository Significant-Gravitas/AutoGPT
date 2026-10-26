"use client";

import { useState } from "react";
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
import {
  CheckmarkSquare02Icon,
  Delete02Icon,
  InformationCircleIcon,
  SquareIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
      className="flex items-center justify-between py-4 pl-3 pr-5 transition-colors data-[selected=true]:bg-surface-hover"
    >
      <div className="flex items-center gap-3">
        <button
          type="button"
          role="checkbox"
          aria-checked={selected}
          aria-label={`Select ${apiKey.name}`}
          onClick={onToggleSelected}
          className={`shrink-0 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-action ${
            selected
              ? "text-foreground-emphasis hover:text-foreground-strong"
              : "text-foreground-tertiary hover:text-foreground-body"
          }`}
        >
          {selected ? (
            <Icon icon={CheckmarkSquare02Icon} size={20} />
          ) : (
            <Icon icon={SquareIcon} size={20} />
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
                    className="shrink-0 rounded text-foreground-tertiary transition-colors hover:text-foreground-body focus:outline-none focus-visible:ring-2 focus-visible:ring-action"
                  >
                    <Icon icon={InformationCircleIcon} size={16} />
                  </button>
                </TooltipTrigger>
                <TooltipContent side="top">View key details</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          <div className="flex items-center gap-3 whitespace-nowrap">
            <Text variant="label" as="span" className="text-foreground-body">
              {maskedKey}
            </Text>
            <Text
              variant="small"
              as="span"
              className="leading-5 text-foreground-tertiary"
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
        className="shrink-0 rounded text-foreground-tertiary transition-colors hover:text-foreground-body focus:outline-none focus-visible:ring-2 focus-visible:ring-action"
      >
        <Icon icon={Delete02Icon} size={20} />
      </button>

      <APIKeyInfoDialog
        open={infoOpen}
        apiKey={apiKey}
        onOpenChange={setInfoOpen}
      />
    </div>
  );
}
