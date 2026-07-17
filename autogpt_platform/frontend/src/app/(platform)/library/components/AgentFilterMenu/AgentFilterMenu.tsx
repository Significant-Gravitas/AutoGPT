"use client";

import type { SelectOption } from "@/components/atoms/Select/Select";
import { Select } from "@/components/atoms/Select/Select";
import { FunnelIcon } from "@phosphor-icons/react";
import type { AgentStatusFilter, FleetSummary } from "../../types";

interface Props {
  value: AgentStatusFilter;
  onChange: (value: AgentStatusFilter) => void;
  summary: FleetSummary;
}

function buildOptions(summary: FleetSummary): SelectOption[] {
  return [
    { value: "all", label: "All Agents" },
    { value: "running", label: `Running (${summary.running})` },
    { value: "attention", label: `Needs Attention (${summary.error})` },
    { value: "listening", label: `Listening (${summary.listening})` },
    { value: "scheduled", label: `Scheduled (${summary.scheduled})` },
    { value: "idle", label: `Idle / Stale (${summary.idle})` },
    { value: "healthy", label: "Healthy" },
  ];
}

export function AgentFilterMenu({ value, onChange, summary }: Props) {
  function handleChange(val: string) {
    onChange(val as AgentStatusFilter);
  }

  const options = buildOptions(summary);

  return (
    <div className="flex items-center" data-testid="agent-filter-dropdown">
      <span className="hidden whitespace-nowrap text-sm text-zinc-500 sm:inline">
        filter
      </span>
      <FunnelIcon className="ml-1 h-4 w-4 sm:hidden" />
      <Select
        id="agent-status-filter"
        label="Filter agents"
        hideLabel
        value={value}
        onValueChange={handleChange}
        options={options}
        size="small"
        className="ml-1 w-fit border-none !bg-transparent text-sm underline underline-offset-4 shadow-none"
        wrapperClassName="mb-0"
      />
    </div>
  );
}
