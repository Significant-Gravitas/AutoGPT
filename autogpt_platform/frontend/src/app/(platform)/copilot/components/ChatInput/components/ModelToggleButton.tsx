"use client";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ToggleChip } from "@/components/atoms/ToggleChip/ToggleChip";
import { cn } from "@/lib/utils";
import { CpuIcon } from "@hugeicons/core-free-icons";
import type { CopilotLlmModel } from "../../../store";

interface Props {
  model: CopilotLlmModel;
  onToggle: () => void;
  /** "pill" is the pre-brain-dump footer style; "chip" lives in the tray. */
  variant?: "chip" | "pill";
}

export function ModelToggleButton({
  model,
  onToggle,
  variant = "chip",
}: Props) {
  const isAdvanced = model === "advanced";
  const tooltip = isAdvanced
    ? "Using the highest-capability model (click to switch to Balanced)."
    : "Using the balanced default model (click to switch to Advanced).";
  const ariaLabel = isAdvanced
    ? "Switch to Balanced model"
    : "Switch to Advanced model";

  if (variant === "pill") {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            aria-pressed={isAdvanced}
            onClick={onToggle}
            className={cn(
              "inline-flex h-9 items-center justify-center gap-1 rounded-full border border-neutral-200 bg-white px-2.5 text-xs font-medium shadow-sm transition-colors hover:bg-neutral-50",
              isAdvanced
                ? "text-emerald-500"
                : "text-sky-600 hover:text-sky-700",
            )}
            aria-label={ariaLabel}
          >
            <Icon icon={CpuIcon} size={14} />
            <span className="hidden sm:inline">
              {isAdvanced ? "Advanced" : "Balanced"}
            </span>
          </button>
        </TooltipTrigger>
        <TooltipContent>{tooltip}</TooltipContent>
      </Tooltip>
    );
  }

  return (
    <ToggleChip
      icon={<Icon icon={CpuIcon} size={14} />}
      label={isAdvanced ? "Advanced" : "Balanced"}
      tooltip={tooltip}
      ariaLabel={ariaLabel}
      pressed={isAdvanced}
      onToggle={onToggle}
      className="sm:min-w-[5.75rem]"
    />
  );
}
