"use client";

import React from "react";
import { Input } from "@/components/atoms/Input/Input";
import { Select } from "@/components/atoms/Select/Select";
import {
  CUSTOM_INTERVAL_MAX,
  clampCustomIntervalValue,
} from "@/lib/cron-expression-utils";

export function CustomInterval({
  value,
  onChange,
}: {
  value: { value: number; unit: "minutes" | "hours" | "days" };
  onChange: (v: { value: number; unit: "minutes" | "hours" | "days" }) => void;
}) {
  return (
    <div className="flex items-end gap-3">
      <Input
        id="custom-interval-value"
        label="Every"
        type="number"
        min={1}
        max={CUSTOM_INTERVAL_MAX[value.unit]}
        value={value.value}
        onChange={(e) =>
          onChange({
            ...value,
            value: clampCustomIntervalValue(
              value.unit,
              parseInt(e.target.value),
            ),
          })
        }
        className="max-w-24"
        size="small"
      />
      <Select
        id="custom-interval-unit"
        label="Interval"
        size="small"
        value={value.unit}
        onValueChange={(v) => {
          const unit = v as "minutes" | "hours" | "days";
          onChange({
            unit,
            value: clampCustomIntervalValue(unit, value.value),
          });
        }}
        options={[
          { label: "Minutes", value: "minutes" },
          { label: "Hours", value: "hours" },
          { label: "Days", value: "days" },
        ]}
        className="max-w-40"
      />
    </div>
  );
}
