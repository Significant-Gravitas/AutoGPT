"use client";

import { cn } from "@/lib/utils";
import { Cpu } from "@phosphor-icons/react";
import type { CopilotLlmModel } from "../../../store";

interface Props {
  model: CopilotLlmModel;
  onToggle: () => void;
}

export function ModelToggleButton({ model, onToggle }: Props) {
  const isAdvanced = model === "advanced";
  return (
    <button
      type="button"
      aria-pressed={isAdvanced}
      onClick={onToggle}
      className={cn(
        "inline-flex h-9 items-center justify-center gap-1 rounded-full border border-neutral-200 bg-white px-2.5 text-xs font-medium shadow-sm transition-colors hover:bg-neutral-50",
        isAdvanced ? "text-sky-900" : "text-neutral-500 hover:text-neutral-700",
      )}
      aria-label={
        isAdvanced ? "Switch to Standard model" : "Switch to Advanced model"
      }
      title={
        isAdvanced
          ? "Advanced model — highest capability (click to switch to Standard)"
          : "Standard model — click to switch to Advanced"
      }
    >
      <Cpu size={14} />
      <span className="hidden sm:inline">
        {isAdvanced ? "Advanced" : "Standard"}
      </span>
    </button>
  );
}
