"use client";

import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import { ArrowRightIcon } from "@phosphor-icons/react";
import NextLink from "next/link";
import { StatusBadge } from "@/app/(platform)/library/components/StatusBadge/StatusBadge";
import type { AgentStatus } from "@/app/(platform)/library/types";

export interface PulseChipData {
  id: string;
  name: string;
  status: AgentStatus;
  shortMessage: string;
}

interface Props {
  chips: PulseChipData[];
  onChipClick?: (prompt: string) => void;
}

export function PulseChips({ chips, onChipClick }: Props) {
  if (chips.length === 0) return null;

  return (
    <div className="mb-6">
      <div className="mb-3 flex items-center justify-between">
        <Text variant="small-medium" className="text-zinc-600">
          What&apos;s happening with your agents
        </Text>
        <NextLink
          href="/library"
          className="flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-700"
        >
          View all <ArrowRightIcon size={12} />
        </NextLink>
      </div>
      <div className="flex flex-wrap gap-2">
        {chips.map((chip) => (
          <PulseChip key={chip.id} chip={chip} onClick={onChipClick} />
        ))}
      </div>
    </div>
  );
}

interface ChipProps {
  chip: PulseChipData;
  onClick?: (prompt: string) => void;
}

function PulseChip({ chip, onClick }: ChipProps) {
  function handleClick() {
    const prompt = buildChipPrompt(chip);
    onClick?.(prompt);
  }

  return (
    <button
      type="button"
      onClick={handleClick}
      className="flex items-center gap-2 rounded-medium border border-zinc-100 bg-white px-3 py-2 text-left transition-all hover:border-zinc-200 hover:shadow-sm"
    >
      <StatusBadge status={chip.status} />
      <div className="min-w-0">
        <Text variant="small-medium" className="truncate text-zinc-900">
          {chip.name}
        </Text>
        <Text variant="xsmall" className="truncate text-zinc-500">
          {chip.shortMessage}
        </Text>
      </div>
    </button>
  );
}

function buildChipPrompt(chip: PulseChipData): string {
  switch (chip.status) {
    case "error":
      return `What happened with ${chip.name}? It has an error — can you check?`;
    case "running":
      return `Give me a status update on ${chip.name} — what has it done so far?`;
    case "idle":
      return `${chip.name} hasn't run recently. Should I keep it or update and re-run it?`;
    default:
      return `Tell me about ${chip.name} — what's its current status?`;
  }
}
