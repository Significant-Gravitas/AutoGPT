"use client";

import React, { useEffect, useMemo, useState } from "react";
import { Select } from "@/components/atoms/Select/Select";

export function TimeAt({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) {
  const [hour12, setHour12] = useState<string>("9");
  const [minute, setMinute] = useState<string>("00");
  const [ampm, setAmPm] = useState<"AM" | "PM">("AM");

  // Parse incoming 24h value → local 12h selects
  useEffect(() => {
    const [hStr, mStr] = (value || "09:00").split(":");
    const h24 = Math.max(0, Math.min(23, parseInt(hStr || "9", 10) || 9));
    const m = Math.max(0, Math.min(59, parseInt(mStr || "0", 10) || 0));
    const isPm = h24 >= 12;
    const h12 = h24 % 12 === 0 ? 12 : h24 % 12;
    setHour12(String(h12));
    setMinute(m.toString().padStart(2, "0"));
    setAmPm(isPm ? "PM" : "AM");
  }, [value]);

  const hourOptions = useMemo(
    () => Array.from({ length: 12 }, (_, i) => String(i + 1)),
    [],
  );
  const minuteOptions = useMemo(
    () =>
      Array.from({ length: 12 }, (_, i) => (i * 5).toString().padStart(2, "0")),
    [],
  );

  function emit(h12Str: string, mStr: string, meridiem: "AM" | "PM") {
    const h12Num = Math.max(
      1,
      Math.min(12, parseInt(h12Str || "12", 10) || 12),
    );
    const mNum = Math.max(0, Math.min(59, parseInt(mStr || "0", 10) || 0));
    let h24 = h12Num % 12;
    if (meridiem === "PM") h24 += 12;
    const next = `${h24.toString().padStart(2, "0")}:${mNum
      .toString()
      .padStart(2, "0")}`;
    onChange(next);
  }

  return (
    <div className="flex items-end gap-2">
      <div className="relative">
        <label className="mb-1 block text-xs font-medium text-zinc-700">
          At
        </label>
        <div className="flex items-center gap-2">
          <Select
            id="time-hour"
            label=""
            size="small"
            value={hour12}
            onValueChange={(v) => {
              setHour12(v);
              emit(v, minute, ampm);
            }}
            options={hourOptions.map((h) => ({ label: h, value: h }))}
            className="max-w-20"
          />
          <Select
            id="time-minute"
            label=""
            size="small"
            value={minute}
            onValueChange={(v) => {
              setMinute(v);
              emit(hour12, v, ampm);
            }}
            options={minuteOptions.map((m) => ({ label: m, value: m }))}
            className="max-w-24"
          />
          <Select
            id="time-meridiem"
            label=""
            size="small"
            value={ampm}
            onValueChange={(v) => {
              const mer = (v as "AM" | "PM") || "AM";
              setAmPm(mer);
              emit(hour12, minute, mer);
            }}
            options={[
              { label: "AM", value: "AM" },
              { label: "PM", value: "PM" },
            ]}
            className="max-w-24"
          />
        </div>
      </div>
    </div>
  );
}
