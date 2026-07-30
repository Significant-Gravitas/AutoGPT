"use client";

import { CpuIcon } from "@phosphor-icons/react";
import type { CopilotLlmModel } from "../../../store";
import { ToggleChip } from "./ToggleChip";

interface Props {
  model: CopilotLlmModel;
  onToggle: () => void;
}

export function ModelToggleButton({ model, onToggle }: Props) {
  const isAdvanced = model === "advanced";

  return (
    <ToggleChip
      icon={<CpuIcon size={14} />}
      label={isAdvanced ? "Advanced" : "Balanced"}
      tooltip={
        isAdvanced
          ? "Using the highest-capability model (click to switch to Balanced)."
          : "Using the balanced default model (click to switch to Advanced)."
      }
      ariaLabel={
        isAdvanced ? "Switch to Balanced model" : "Switch to Advanced model"
      }
      pressed={isAdvanced}
      onToggle={onToggle}
      className="sm:min-w-[5.75rem]"
    />
  );
}
