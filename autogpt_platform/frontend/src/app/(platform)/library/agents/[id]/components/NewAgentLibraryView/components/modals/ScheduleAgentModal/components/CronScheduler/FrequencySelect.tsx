"use client";

import React from "react";
import { Select } from "@/components/atoms/Select/Select";

type CronFrequency =
  | "hourly"
  | "daily"
  | "weekly"
  | "monthly"
  | "yearly"
  | "custom"
  | "every minute";

export function FrequencySelect({
  value,
  onChange,
  selectedMinute,
  onMinuteChange,
}: {
  value: CronFrequency;
  onChange: (v: CronFrequency) => void;
  selectedMinute: string;
  onMinuteChange: (v: string) => void;
}) {
  return (
    <>
      <Select
        id="repeat"
        label="Repeats"
        size="small"
        value={value}
        onValueChange={(v) => onChange(v as CronFrequency)}
        options={[
          { label: "Every Hour", value: "hourly" },
          { label: "Daily", value: "daily" },
          { label: "Weekly", value: "weekly" },
          { label: "Monthly", value: "monthly" },
          { label: "Yearly", value: "yearly" },
          { label: "Custom", value: "custom" },
        ]}
        className="max-w-80"
      />
      {value === "hourly" && (
        <Select
          id="at-minute"
          label="At minute"
          size="small"
          value={selectedMinute}
          onValueChange={(v) => onMinuteChange(v)}
          options={["0", "15", "30", "45"].map((m) => ({ label: m, value: m }))}
          className="max-w-32"
        />
      )}
    </>
  );
}
