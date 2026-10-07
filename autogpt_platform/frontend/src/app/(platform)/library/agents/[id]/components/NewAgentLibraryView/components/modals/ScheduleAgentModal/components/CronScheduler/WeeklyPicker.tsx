"use client";

import React from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { MultiToggle } from "@/components/molecules/MultiToggle/MultiToggle";

const weekDays = [
  { label: "Su", value: 0 },
  { label: "Mo", value: 1 },
  { label: "Tu", value: 2 },
  { label: "We", value: 3 },
  { label: "Th", value: 4 },
  { label: "Fr", value: 5 },
  { label: "Sa", value: 6 },
];

export function WeeklyPicker({
  values,
  onChange,
}: {
  values: number[];
  onChange: (v: number[]) => void;
}) {
  function toggleAll() {
    if (values.length === weekDays.length) onChange([]);
    else onChange(weekDays.map((d) => d.value));
  }
  function setWeekdays() {
    onChange([1, 2, 3, 4, 5]);
  }
  function setWeekends() {
    onChange([0, 6]);
  }
  const items = weekDays.map((d) => ({
    value: String(d.value),
    label: d.label,
  }));
  const selectedValues = values.map((v) => String(v));

  return (
    <div className="mb-8 space-y-3">
      <Text variant="body-medium" as="span" className="text-black">
        Repeats on
      </Text>
      <div className="flex flex-wrap gap-2">
        <Button type="button" variant="outline" size="md" onClick={toggleAll}>
          Select all
        </Button>
        <Button type="button" variant="outline" size="md" onClick={setWeekdays}>
          Weekdays
        </Button>
        <Button type="button" variant="outline" size="md" onClick={setWeekends}>
          Weekends
        </Button>
      </div>
      <MultiToggle
        items={items}
        selectedValues={selectedValues}
        onChange={(sv) => onChange(sv.map((s) => parseInt(s)))}
        aria-label="Select days of week"
      />
    </div>
  );
}
