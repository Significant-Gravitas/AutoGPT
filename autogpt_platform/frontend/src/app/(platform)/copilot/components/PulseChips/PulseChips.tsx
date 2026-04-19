"use client";

import { Text } from "@/components/atoms/Text/Text";
import {
  ArrowRightIcon,
  EyeIcon,
  ChatCircleDotsIcon,
} from "@phosphor-icons/react";
import NextLink from "next/link";
import { StatusBadge } from "@/app/(platform)/library/components/StatusBadge/StatusBadge";
import styles from "./PulseChips.module.css";
import type { PulseChipData } from "./types";

interface Props {
  chips: PulseChipData[];
  onChipClick?: (prompt: string) => void;
}

export function PulseChips({ chips, onChipClick }: Props) {
  if (chips.length === 0) return null;

  return (
    <div
      className={`${styles.glassPanel} mx-[0.6875rem] mb-5 rounded-large p-5`}
    >
      <div className="mb-3 flex items-center gap-3">
        <Text variant="body-medium" className="text-zinc-600">
          What&apos;s happening with your agents
        </Text>
        <NextLink
          href="/library"
          className="flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-700"
        >
          View all <ArrowRightIcon size={12} />
        </NextLink>
      </div>
      <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-300">
        {chips.map((chip) => (
          <PulseChip key={chip.id} chip={chip} onAsk={onChipClick} />
        ))}
      </div>
    </div>
  );
}

interface ChipProps {
  chip: PulseChipData;
  onAsk?: (prompt: string) => void;
}

function PulseChip({ chip, onAsk }: ChipProps) {
  function handleAsk() {
    const prompt = buildChipPrompt(chip);
    onAsk?.(prompt);
  }

  return (
    <div
      className={`${styles.chip} relative flex w-[15rem] shrink-0 flex-col items-start gap-2 rounded-medium border border-zinc-100 bg-white px-3 py-2`}
    >
      <div className={`${styles.chipContent} w-full text-left`}>
        {chip.priority === "success" ? (
          <span className="inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium text-emerald-600">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
            Completed
          </span>
        ) : (
          <StatusBadge status={chip.status} />
        )}
        <div className="mt-2 min-w-0">
          <Text variant="small-medium" className="truncate text-zinc-900">
            {chip.name}
          </Text>
          <Text variant="small" className="truncate text-zinc-500">
            {chip.shortMessage}
          </Text>
        </div>
      </div>
      <div
        className={`${styles.chipActions} flex items-center justify-center gap-1.5 rounded-b-medium px-3 py-1.5`}
      >
        <NextLink
          href={`/library/agents/${chip.agentID}`}
          className="flex items-center gap-1 rounded-md px-2 py-1 text-xs text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-700"
        >
          <EyeIcon size={14} />
          See
        </NextLink>
        <button
          type="button"
          onClick={handleAsk}
          className="flex items-center gap-1 rounded-md px-2 py-1 text-xs text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-700"
        >
          <ChatCircleDotsIcon size={14} />
          Ask
        </button>
      </div>
    </div>
  );
}

function buildChipPrompt(chip: PulseChipData): string {
  if (chip.priority === "success") {
    return `${chip.name} just finished a run — can you summarize what it did?`;
  }
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
