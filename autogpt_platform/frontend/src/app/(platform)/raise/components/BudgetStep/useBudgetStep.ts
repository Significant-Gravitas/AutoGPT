import { useState } from "react";
import { parseBudget } from "../KitStep/helpers";

interface Args {
  onSubmit: (credits: number) => void;
}

export function useBudgetStep({ onSubmit }: Args) {
  const [weeklyBudget, setWeeklyBudget] = useState<number | null>(null);
  const [customAmount, setCustomAmount] = useState("");
  const parsedCustom = parseBudget(customAmount);

  function selectPreset(credits: number) {
    setWeeklyBudget(credits);
    setCustomAmount("");
    onSubmit(credits);
  }

  function changeCustomAmount(value: string) {
    setCustomAmount(value);
    const parsed = parseBudget(value);
    if (parsed !== null) setWeeklyBudget(parsed);
  }

  function submitCustom() {
    if (parsedCustom === null) return;
    onSubmit(parsedCustom);
  }

  return {
    weeklyBudget,
    customAmount,
    canSubmitCustom: parsedCustom !== null,
    selectPreset,
    changeCustomAmount,
    submitCustom,
  };
}
