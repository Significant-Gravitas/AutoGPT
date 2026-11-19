"use client";

import { DateTimeInput } from "@/components/atoms/DateTimeInput/DateTimeInput";
import { useNowMinute } from "@/hooks/useNowMinute";
import { fromLocalInputValue, toLocalInputValue } from "../../../../schedule";
import { ChoiceMenu } from "../../../ChoiceMenu/ChoiceMenu";
import {
  defaultStartOn,
  DIALOG_QUICK_STARTS,
  quickLabel,
  startChipLabel,
  type StartChoice,
} from "../../startChoice";

const NOW = "now";
const PICK = "pick";

interface Props {
  value: StartChoice | null;
  onChange: (start: StartChoice | null) => void;
}

function startOfToday(now: number) {
  const day = new Date(now);
  day.setHours(0, 0, 0, 0);
  return day;
}

/** When the task starts: now (the default), a quick pick shown with the
 *  time it means (worked out again when the task is sent), or a date and
 *  time from today on. */
export function StartPicker({ value, onChange }: Props) {
  const now = useNowMinute();
  const choices = [
    { value: NOW, label: "Now" },
    ...DIALOG_QUICK_STARTS.map((start) => ({
      value: start.label,
      label: quickLabel(start.label, now),
    })),
    { value: PICK, label: "Pick a date and time" },
  ];
  const picked = value?.kind === "quick" ? value.label : value ? PICK : NOW;

  function pick(next: string) {
    if (next === NOW) return onChange(null);
    if (next === PICK) {
      if (value?.kind !== "at") onChange({ kind: "picking" });
      return;
    }
    onChange({ kind: "quick", label: next });
  }

  // A day picked for the first time comes at midnight: start it at the
  // next round hour today, or 09:00 on a later day.
  function pickDate(next: string | undefined) {
    const at = fromLocalInputValue(next);
    if (!at) return onChange({ kind: "picking" });
    const first = value?.kind !== "at";
    onChange({ kind: "at", at: first ? defaultStartOn(at, Date.now()) : at });
  }

  return (
    <>
      <ChoiceMenu
        name="Start"
        value={picked}
        choices={choices}
        isActive={value !== null}
        onChange={pick}
        display={startChipLabel(value, now)}
      />
      {value?.kind === "picking" || value?.kind === "at" ? (
        <DateTimeInput
          size="small"
          aria-label="Start time"
          value={value.kind === "at" ? toLocalInputValue(value.at) : ""}
          onChange={pickDate}
          minDate={startOfToday(now)}
          placeholder="Pick a date"
          wrapperClassName="min-w-48"
        />
      ) : null}
    </>
  );
}
