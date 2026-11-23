"use client";

import { TimeInput } from "@/components/atoms/TimeInput/TimeInput";
import { ChoiceMenu } from "../../../ChoiceMenu/ChoiceMenu";
import {
  DEFAULT_DAY,
  DEFAULT_TIME,
  REPEAT_KINDS,
  rhythmOf,
  WEEK_DAYS,
  type RepeatChoice,
} from "../../repeat";

const NEVER = "never";
const KIND_CHOICES = [
  { value: NEVER, label: "Doesn't repeat" },
  ...REPEAT_KINDS.map(({ kind, label }) => ({ value: kind, label })),
];
const DAY_CHOICES = WEEK_DAYS.map((day, index) => ({
  value: String(index),
  label: day,
}));

interface Props {
  value: RepeatChoice | null;
  onChange: (repeat: RepeatChoice | null) => void;
}

/** How often the task runs: never (the default), every hour, every day,
 *  weekdays or one day a week, at a time; shown in words. */
export function RepeatPicker({ value, onChange }: Props) {
  function pickKind(next: string) {
    const kind = REPEAT_KINDS.find((entry) => entry.kind === next)?.kind;
    if (!kind) return onChange(null);
    onChange({
      kind,
      time: value?.time ?? DEFAULT_TIME,
      day: value?.day ?? DEFAULT_DAY,
    });
  }

  return (
    <>
      <ChoiceMenu
        name="Repeat"
        value={value?.kind ?? NEVER}
        choices={KIND_CHOICES}
        isActive={value !== null}
        onChange={pickKind}
        display={value ? rhythmOf(value) : "Doesn't repeat"}
      />
      {value?.kind === "weekly" ? (
        <ChoiceMenu
          name="Day"
          value={String(value.day)}
          choices={DAY_CHOICES}
          isActive
          onChange={(day) => onChange({ ...value, day: Number(day) })}
        />
      ) : null}
      {value && value.kind !== "hourly" ? (
        <TimeInput
          size="small"
          aria-label="Repeat time"
          value={value.time}
          onChange={(time) => onChange({ ...value, time })}
          wrapperClassName="mb-0 w-28"
        />
      ) : null}
    </>
  );
}
