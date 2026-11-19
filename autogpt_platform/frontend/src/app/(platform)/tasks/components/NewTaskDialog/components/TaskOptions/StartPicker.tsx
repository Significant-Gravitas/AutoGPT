"use client";

import { DateTimeInput } from "@/components/atoms/DateTimeInput/DateTimeInput";
import { useNowMinute } from "@/hooks/useNowMinute";
import { useState } from "react";
import {
  formatStartTime,
  fromLocalInputValue,
  QUICK_STARTS,
  toLocalInputValue,
} from "../../../../schedule";
import { ChoiceMenu } from "../../../ChoiceMenu/ChoiceMenu";

const NOW = "now";
const PICK = "pick";
/** "In 5 min" is the sidebar's; the dialog offers the hour and tomorrow. */
const STARTS = QUICK_STARTS.filter((start) => start.label !== "In 5 min");
const CHOICES = [
  { value: NOW, label: "Now" },
  ...STARTS.map((start) => ({ value: start.label, label: start.label })),
  { value: PICK, label: "Pick a date and time" },
];

interface Props {
  value: Date | null;
  onChange: (startAt: Date | null) => void;
}

/** When the task starts: now (the default), a quick pick, or a date and
 *  time. */
export function StartPicker({ value, onChange }: Props) {
  const [choice, setChoice] = useState(NOW);
  const now = useNowMinute(value !== null);
  const picked = value && choice !== NOW ? choice : NOW;

  function pick(next: string) {
    setChoice(next);
    const quick = STARTS.find((start) => start.label === next);
    if (quick) onChange(quick.at(Date.now()));
    else if (next === NOW) onChange(null);
  }

  return (
    <>
      <ChoiceMenu
        name="Start"
        value={picked}
        choices={CHOICES}
        isActive={value !== null}
        onChange={pick}
        display={value ? `Starts ${formatStartTime(value, now)}` : "Starts now"}
      />
      {choice === PICK ? (
        <DateTimeInput
          size="small"
          aria-label="Start time"
          value={value ? toLocalInputValue(value) : ""}
          onChange={(next) => onChange(fromLocalInputValue(next))}
          wrapperClassName="min-w-48"
        />
      ) : null}
    </>
  );
}
