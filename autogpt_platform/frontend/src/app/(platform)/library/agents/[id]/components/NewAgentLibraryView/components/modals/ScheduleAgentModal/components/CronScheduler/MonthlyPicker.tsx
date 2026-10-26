"use client";

import React from "react";
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
      <Text variant="body-medium" as="span" className="text-foreground-base">
        Days of Month
      </Text>
      <div className="flex gap-2">
        <button
          type="button"
          // TODO(tokens): no fitting role for border-zinc-700 #505057; #505057 is foreground-body, a text colour
          className={`h-[2.25rem] rounded-full border border-zinc-700 px-4 py-2 text-sm font-medium leading-4 text-foreground-base hover:bg-surface-hover`}
          onClick={allDays}
        >
          All Days
        </button>
        <button
          type="button"
          // TODO(tokens): no fitting role for border-zinc-700 #505057; #505057 is foreground-body, a text colour
          className={`h-[2.25rem] rounded-full border border-zinc-700 px-4 py-2 text-sm font-medium leading-4 text-foreground-base hover:bg-surface-hover`}
          onClick={customize}
        >
          Customize
        </button>
        <button
          type="button"
          // TODO(tokens): no fitting role for border-zinc-700 #505057; #505057 is foreground-body, a text colour
          className="h-[2.25rem] rounded-full border border-zinc-700 px-4 py-2 text-sm font-medium leading-4 text-foreground-base hover:bg-surface-hover"
          onClick={() => onChange([15])}
        >
          15th
        </button>
        <button
          type="button"
          // TODO(tokens): no fitting role for border-zinc-700 #505057; #505057 is foreground-body, a text colour
          className="h-[2.25rem] rounded-full border border-zinc-700 px-4 py-2 text-sm font-medium leading-4 text-foreground-base hover:bg-surface-hover"
          onClick={() => onChange([31])}
        >
          Last Day
        </button>
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
