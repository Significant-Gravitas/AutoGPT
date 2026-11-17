import { useState } from "react";
import {
  type CustomIntervalUnit,
  getCustomIntervalError,
  parseCustomInterval,
} from "@/lib/cron-expression-utils";

export interface CustomIntervalDraft {
  value: string;
  unit: CustomIntervalUnit;
}

interface Args {
  interval: CustomIntervalDraft;
  onChange: (interval: CustomIntervalDraft) => void;
}

function errorOf(interval: CustomIntervalDraft) {
  return getCustomIntervalError(
    interval.unit,
    parseCustomInterval(interval.value),
  );
}

// The field keeps exactly what was typed. An out-of-range value is only
// flagged once the user leaves the field or changes the unit; an edit that
// makes it valid clears the message straight away.
export function useCustomIntervalField({ interval, onChange }: Args) {
  const [isErrorShown, setIsErrorShown] = useState(false);
  const error = errorOf(interval);

  function handleValueChange(e: React.ChangeEvent<HTMLInputElement>) {
    const next = { ...interval, value: e.target.value };
    if (!errorOf(next)) setIsErrorShown(false);
    onChange(next);
  }

  function handleBlur() {
    setIsErrorShown(error !== undefined);
  }

  function handleUnitChange(unit: string) {
    const next = { ...interval, unit: unit as CustomIntervalUnit };
    setIsErrorShown(errorOf(next) !== undefined);
    onChange(next);
  }

  return {
    error: isErrorShown ? error : undefined,
    handleValueChange,
    handleBlur,
    handleUnitChange,
  };
}
