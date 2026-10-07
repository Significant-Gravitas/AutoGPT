"use client";

import React from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { MultiToggle } from "@/components/molecules/MultiToggle/MultiToggle";

const months = [
  { label: "Jan", value: 1 },
  { label: "Feb", value: 2 },
  { label: "Mar", value: 3 },
  { label: "Apr", value: 4 },
  { label: "May", value: 5 },
  { label: "Jun", value: 6 },
  { label: "Jul", value: 7 },
  { label: "Aug", value: 8 },
  { label: "Sep", value: 9 },
  { label: "Oct", value: 10 },
  { label: "Nov", value: 11 },
  { label: "Dec", value: 12 },
];

export function YearlyPicker({
  values,
  onChange,
}: {
  values: number[];
  onChange: (v: number[]) => void;
}) {
  function toggleAll() {
    if (values.length === months.length) onChange([]);
    else onChange(months.map((m) => m.value));
  }
  const items = months.map((m) => ({ value: String(m.value), label: m.label }));
  const selected = values.map((v) => String(v));

  return (
    <div className="mb-6 space-y-2">
      <Text variant="body-medium" as="span" className="text-black">
        Months
      </Text>
      <div className="flex gap-2">
        <Button
          type="button"
          variant="outline"
          size="small"
          onClick={toggleAll}
        >
          {values.length === months.length ? "Deselect All" : "Select All"}
        </Button>
      </div>
      <MultiToggle
        items={items}
        selectedValues={selected}
        onChange={(sv) => onChange(sv.map((s) => parseInt(s)))}
        aria-label="Select months"
      />
    </div>
  );
}
