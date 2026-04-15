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
        "inline-flex min-h-11 min-w-11 items-center justify-center gap-1 rounded-md px-2 py-1 text-xs font-medium transition-colors",
        isAdvanced
          ? "bg-sky-100 text-sky-900 hover:bg-sky-200"
          : "text-neutral-500 hover:bg-neutral-100 hover:text-neutral-700",
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
      {isAdvanced && "Advanced"}
    </button>
  );
}
