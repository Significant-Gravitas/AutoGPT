"use client";

import { useState } from "react";
import type { PulseChipData } from "./PulseChips";
import type { AgentStatus } from "@/app/(platform)/library/types";

/**
 * Provides a prioritised list of pulse chips for the Home empty state.
 * Errors → running → stale, max 5 chips.
 *
 * TODO: Replace with real API data from `GET /agents/summary` or similar.
 */
export function usePulseChips(): PulseChipData[] {
  const [chips] = useState<PulseChipData[]>(() => MOCK_CHIPS);
  return chips;
}

const MOCK_CHIPS: PulseChipData[] = [
  {
    id: "chip-1",
    name: "Lead Finder",
    status: "error" as AgentStatus,
    shortMessage: "API rate limit hit",
  },
  {
    id: "chip-2",
    name: "CEO Finder",
    status: "running" as AgentStatus,
    shortMessage: "72% complete",
  },
  {
    id: "chip-3",
    name: "Cart Recovery",
    status: "idle" as AgentStatus,
    shortMessage: "No runs in 3 weeks",
  },
  {
    id: "chip-4",
    name: "Social Collector",
    status: "listening" as AgentStatus,
    shortMessage: "Waiting for trigger",
  },
];
