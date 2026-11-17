"use client";

import React from "react";
import { Input } from "@/components/atoms/Input/Input";
import { Select } from "@/components/atoms/Select/Select";
import { CUSTOM_INTERVAL_MAX } from "@/lib/cron-expression-utils";
import {
  type CustomIntervalDraft,
  useCustomIntervalField,
} from "@/lib/hooks/useCustomIntervalField";

interface Props {
  value: CustomIntervalDraft;
  onChange: (v: CustomIntervalDraft) => void;
}

export function CustomInterval({ value, onChange }: Props) {
  const { error, handleValueChange, handleBlur, handleUnitChange } =
    useCustomIntervalField({ interval: value, onChange });

  return (
    <div className="flex items-end gap-3">
      <Input
        id="custom-interval-value"
        label="Every"
        type="number"
        min={1}
        max={CUSTOM_INTERVAL_MAX[value.unit]}
        value={value.value}
        onChange={handleValueChange}
        onBlur={handleBlur}
        error={error}
        aria-invalid={error ? true : undefined}
        className="max-w-24"
        size="small"
      />
      <Select
        id="custom-interval-unit"
        label="Interval"
        size="small"
        value={value.unit}
        onValueChange={handleUnitChange}
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
