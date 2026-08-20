import { useState } from "react";
import { parseCredits } from "../KitStep/helpers";

interface Args {
  onSubmit: (credits: number) => void;
}

export function useBudgetStep({ onSubmit }: Args) {
  const [weeklyBudget, setWeeklyBudget] = useState<number | null>(null);
  const [customCredits, setCustomCredits] = useState("");
  const parsedCustom = parseCredits(customCredits);

  function selectPreset(credits: number) {
    setWeeklyBudget(credits);
    setCustomCredits("");
    onSubmit(credits);
  }

  function changeCustomCredits(value: string) {
    setCustomCredits(value);
    const parsed = parseCredits(value);
    if (parsed !== null) setWeeklyBudget(parsed);
  }

  function submitCustom() {
    if (parsedCustom === null) return;
    onSubmit(parsedCustom);
  }

  return {
    weeklyBudget,
    customCredits,
    canSubmitCustom: parsedCustom !== null,
    selectPreset,
    changeCustomCredits,
    submitCustom,
  };
}
