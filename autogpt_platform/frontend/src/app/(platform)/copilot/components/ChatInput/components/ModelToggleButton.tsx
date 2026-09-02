"use client";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { CpuIcon } from "@hugeicons/core-free-icons";
import type { CopilotLlmModel } from "../../../store";

interface Props {
  model: CopilotLlmModel;
  onToggle: () => void;
}

export function ModelToggleButton({ model, onToggle }: Props) {
  const isAdvanced = model === "advanced";
  const tooltip = isAdvanced
    ? "Using the highest-capability model (click to switch to Balanced)."
    : "Using the balanced default model (click to switch to Advanced).";
  const ariaLabel = isAdvanced
    ? "Switch to Balanced model"
    : "Switch to Advanced model";

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-pressed={isAdvanced}
          onClick={onToggle}
          className={cn(
            "inline-flex h-8 w-8 items-center justify-center rounded-full transition-colors hover:bg-zinc-100",
            // Model tier drives cost and quality, and a tooltip never opens on
            // touch — so the raised tier has to be visible on the icon itself.
            isAdvanced
              ? "text-emerald-600 hover:text-emerald-700"
              : "text-zinc-500 hover:text-zinc-700",
          )}
          aria-label={ariaLabel}
        >
          <Icon icon={CpuIcon} size={16} />
        </button>
      </TooltipTrigger>
      <TooltipContent>{tooltip}</TooltipContent>
    </Tooltip>
  );
}
