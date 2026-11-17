import { useState } from "react";
import {
  CUSTOM_INTERVAL_MAX,
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

function isAboveMax(interval: CustomIntervalDraft) {
  return (
    parseCustomInterval(interval.value) > CUSTOM_INTERVAL_MAX[interval.unit]
  );
}

function isCustomIntervalUnit(unit: string): unit is CustomIntervalUnit {
  return unit === "minutes" || unit === "hours" || unit === "days";
}

// The field keeps exactly what was typed. A value above the unit's range is
// flagged at once, since more digits can't fix it; an empty or partial value
// only once the user leaves the field or changes the unit. An edit that makes
// it valid clears the message straight away.
export function useCustomIntervalField({ interval, onChange }: Args) {
  const [isErrorShown, setIsErrorShown] = useState(false);
  const error = errorOf(interval);

  function handleValueChange(e: React.ChangeEvent<HTMLInputElement>) {
    const next = { ...interval, value: e.target.value };
    if (!errorOf(next)) setIsErrorShown(false);
    else if (isAboveMax(next)) setIsErrorShown(true);
    onChange(next);
  }

  function handleBlur() {
    setIsErrorShown(error !== undefined);
  }

  function handleUnitChange(unit: string) {
    if (!isCustomIntervalUnit(unit)) return;
    const next = { ...interval, unit };
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
