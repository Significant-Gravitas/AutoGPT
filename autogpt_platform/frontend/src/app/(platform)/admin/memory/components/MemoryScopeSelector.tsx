"use client";

import type { Expert } from "@/app/api/__generated__/models/expert";
import { Select } from "@/components/atoms/Select/Select";
import { AUTOPILOT_MEMORY_SCOPE } from "./useMemoryScope";

interface Props {
  value: string;
  onValueChange: (value: string) => void;
  experts: Expert[];
  loading: boolean;
  error: unknown;
}

export function MemoryScopeSelector({
  value,
  onValueChange,
  experts,
  loading,
  error,
}: Props) {
  const labelCounts = experts.reduce<Record<string, number>>(
    (counts, expert) => {
      const label = `${expert.name} — ${expert.role.trim() || "Expert"}`;
      counts[label] = (counts[label] ?? 0) + 1;
      return counts;
    },
    {},
  );
  const options = [
    { value: AUTOPILOT_MEMORY_SCOPE, label: "AutoPilot (account memory)" },
    ...experts.map((expert) => {
      const label = `${expert.name} — ${expert.role.trim() || "Expert"}`;
      return {
        value: expert.id,
        label:
          (labelCounts[label] ?? 0) > 1 ? `${label} (${expert.id})` : label,
      };
    }),
  ];

  return (
    <div className="flex flex-col gap-2 rounded-md border bg-white p-3 sm:flex-row sm:items-end sm:justify-between">
      <div className="w-full sm:max-w-sm">
        <Select
          id="memory-scope"
          label="Memory scope"
          value={value}
          onValueChange={onValueChange}
          options={options}
          size="small"
          disabled={loading}
        />
      </div>
      <div aria-live="polite" className="flex flex-col gap-1 text-xs">
        {loading ? <p className="text-gray-500">Loading experts…</p> : null}
        {value !== AUTOPILOT_MEMORY_SCOPE ? (
          <p className="text-gray-500">Expert memory is read-only.</p>
        ) : null}
        {error ? <p className="text-red-700">Failed to load experts.</p> : null}
        {!loading && !error && experts.length === 0 ? (
          <p className="text-gray-500">No experts for this account.</p>
        ) : null}
      </div>
    </div>
  );
}
