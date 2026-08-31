"use client";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ToggleChip } from "@/components/atoms/ToggleChip/ToggleChip";
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
            className="inline-flex h-8 items-center justify-center gap-1.5 rounded-full px-2.5 text-[13px] font-medium text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-700"
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
