"use client";

import React from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { MultiToggle } from "@/components/molecules/MultiToggle/MultiToggle";

export function MonthlyPicker({
  values,
  onChange,
}: {
  values: number[];
  onChange: (v: number[]) => void;
}) {
  function allDays() {
    onChange(Array.from({ length: 31 }, (_, i) => i + 1));
  }
  function customize() {
    onChange([]);
  }
  const items = Array.from({ length: 31 }, (_, i) => ({
    value: String(i + 1),
    label: String(i + 1),
  }));
  const selected = values.map((v) => String(v));

  return (
    <div className="mb-6 space-y-2">
      <Text variant="body-medium" as="span" className="text-black">
        Days of Month
      </Text>
      <div className="flex gap-2">
        <Button type="button" variant="outline" size="md" onClick={allDays}>
          All Days
        </Button>
        <Button type="button" variant="outline" size="md" onClick={customize}>
          Customize
        </Button>
        <Button
          type="button"
          variant="outline"
          size="md"
          onClick={() => onChange([15])}
        >
          15th
        </Button>
        <Button
          type="button"
          variant="outline"
          size="md"
          onClick={() => onChange([31])}
        >
          Last Day
        </Button>
      </div>

      {values.length < 31 && (
        <MultiToggle
          items={items}
          selectedValues={selected}
          onChange={(sv) => onChange(sv.map((s) => parseInt(s)))}
          aria-label="Select days of month"
        />
      )}
    </div>
  );
}
