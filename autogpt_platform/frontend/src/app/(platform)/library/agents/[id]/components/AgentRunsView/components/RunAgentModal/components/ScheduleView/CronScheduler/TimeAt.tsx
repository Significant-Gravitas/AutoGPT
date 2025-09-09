"use client";

import React from "react";
import { Input } from "@/components/atoms/Input/Input";

export function TimeAt({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="relative max-w-32">
      <Input
        id="schedule-time"
        label="At"
        size="small"
        value={value}
        onChange={(e) => onChange((e.target.value || "00:00").trim())}
        placeholder="00:00"
      />
    </div>
  );
}
