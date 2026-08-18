"use client";

import type { Expert } from "@/app/api/__generated__/models/expert";
import { Select } from "@/components/atoms/Select/Select";
import { Text } from "@/components/atoms/Text/Text";
import { AUTOPILOT_SCOPE } from "../helpers";

interface Props {
  scopeExpertID: string | null;
  experts: Expert[];
  onSelect: (expertID: string | null) => void;
}

export function ScopeCard({ scopeExpertID, experts, onSelect }: Props) {
  const options = [
    { value: AUTOPILOT_SCOPE, label: "AutoPilot" },
    ...experts.map((expert) => ({
      value: expert.id,
      label: expert.role ? `${expert.name} — ${expert.role}` : expert.name,
    })),
  ];

  function handleChange(value: string) {
    onSelect(value === AUTOPILOT_SCOPE ? null : value);
  }

  return (
    <div className="flex flex-col gap-3 rounded-[18px] border border-zinc-200 bg-white px-4 py-4 shadow-[0_1px_2px_rgba(15,15,20,0.04)] sm:flex-row sm:items-center sm:justify-between">
      <div className="flex min-w-0 flex-col">
        <Text variant="body-medium" as="span" className="text-textBlack">
          Memory scope
        </Text>
        <Text variant="small" as="span" className="text-zinc-500">
          Choose whose memory you&apos;re looking at. Experts keep their own.
        </Text>
      </div>
      <div className="w-full sm:w-[280px]">
        <Select
          id="memory-scope"
          label="Memory scope"
          hideLabel
          value={scopeExpertID ?? AUTOPILOT_SCOPE}
          onValueChange={handleChange}
          options={options}
        />
      </div>
    </div>
  );
}
