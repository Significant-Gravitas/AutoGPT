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
  const options = [
    { value: AUTOPILOT_MEMORY_SCOPE, label: "AutoPilot (my memory)" },
    ...experts.map((expert) => ({
      value: expert.id,
      label: expert.name,
    })),
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
      {value !== AUTOPILOT_MEMORY_SCOPE ? (
        <p className="text-xs text-gray-500">Expert memory is read-only.</p>
      ) : null}
      {error ? (
        <p className="text-xs text-red-700">Failed to load experts.</p>
      ) : null}
    </div>
  );
}
