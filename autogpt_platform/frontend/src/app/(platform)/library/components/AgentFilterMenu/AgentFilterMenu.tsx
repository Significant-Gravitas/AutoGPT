"use client";

import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/__legacy__/ui/select";
import { FunnelIcon } from "@phosphor-icons/react";
import type { AgentStatusFilter, FleetSummary } from "../../types";

interface Props {
  value: AgentStatusFilter;
  onChange: (value: AgentStatusFilter) => void;
  summary: FleetSummary;
}

export function AgentFilterMenu({ value, onChange, summary }: Props) {
  function handleChange(val: string) {
    onChange(val as AgentStatusFilter);
  }

  return (
    <div className="flex items-center" data-testid="agent-filter-dropdown">
      <span className="hidden whitespace-nowrap text-sm sm:inline">filter</span>
      <Select value={value} onValueChange={handleChange}>
        <SelectTrigger className="ml-1 w-fit space-x-1 border-none px-0 text-sm underline underline-offset-4 shadow-none">
          <FunnelIcon className="h-4 w-4 sm:hidden" />
          <SelectValue placeholder="All Agents" />
        </SelectTrigger>
        <SelectContent>
          <SelectGroup>
            <SelectItem value="all">All Agents</SelectItem>
            <SelectItem value="running">Running ({summary.running})</SelectItem>
            <SelectItem value="attention">
              Needs Attention ({summary.error})
            </SelectItem>
            <SelectItem value="listening">
              Listening ({summary.listening})
            </SelectItem>
            <SelectItem value="scheduled">
              Scheduled ({summary.scheduled})
            </SelectItem>
            <SelectItem value="idle">Idle / Stale ({summary.idle})</SelectItem>
            <SelectItem value="healthy">Healthy</SelectItem>
          </SelectGroup>
        </SelectContent>
      </Select>
    </div>
  );
}
