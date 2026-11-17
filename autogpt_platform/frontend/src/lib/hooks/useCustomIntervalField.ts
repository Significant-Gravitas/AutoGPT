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

// The field keeps exactly what was typed and any invalid value is flagged,
// prefilled ones included, except an empty or partial value the user is still
// typing: that waits until they leave the field or change the unit. A value
// above the unit's range is flagged at once, since more digits can't fix it.
export function useCustomIntervalField({ interval, onChange }: Args) {
  const [isMidEdit, setIsMidEdit] = useState(false);
  const error = errorOf(interval);

  function handleValueChange(e: React.ChangeEvent<HTMLInputElement>) {
    const next = { ...interval, value: e.target.value };
    setIsMidEdit(errorOf(next) !== undefined && !isAboveMax(next));
    onChange(next);
  }

  function handleBlur() {
    setIsMidEdit(false);
  }

  function handleUnitChange(unit: string) {
    if (!isCustomIntervalUnit(unit)) return;
    setIsMidEdit(false);
    onChange({ ...interval, unit });
  }

  return {
    error: isMidEdit ? undefined : error,
    handleValueChange,
    handleBlur,
    handleUnitChange,
  };
}
