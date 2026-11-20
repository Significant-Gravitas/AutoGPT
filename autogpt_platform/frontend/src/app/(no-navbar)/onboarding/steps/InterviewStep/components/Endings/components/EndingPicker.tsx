"use client";

import { Select } from "@/components/atoms/Select/Select";
import { useState } from "react";
import {
  type EndingArm,
  type EndingMode,
  endingArmOf,
  ENDING_NAMES,
  MODE_ARMS,
  readEndingPick,
  saveEndingPick,
} from "../helpers";

const DEFAULT = "default";

interface Props {
  mode: EndingMode;
  onPick?: (arm: EndingArm | null) => void;
}

// Testers pick how the interview ends; remembered per browser.
export function EndingPicker({ mode, onPick }: Props) {
  const [pick, setPick] = useState(() => readEndingPick(mode));

  function handleChange(value: string) {
    const arm = endingArmOf(value);
    setPick(arm);
    saveEndingPick(mode, arm);
    onPick?.(arm);
  }

  return (
    <div className="w-full max-w-56">
      <Select
        id={`onboarding-ending-pick-${mode}`}
        label="Ending (testers)"
        size="small"
        labelVariant="small"
        value={pick ?? DEFAULT}
        onValueChange={handleChange}
        options={[
          { value: DEFAULT, label: "Default" },
          ...MODE_ARMS[mode].map((arm) => ({
            value: arm,
            label: ENDING_NAMES[arm],
          })),
        ]}
      />
    </div>
  );
}
