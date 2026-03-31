"use client";

import { useState } from "react";
import type { FleetSummary } from "../types";

/**
 * Returns fleet-wide summary counts for the Agent Briefing Panel.
 *
 * TODO: Replace with a real `GET /agents/summary` API call once available.
 * For now, returns deterministic mock data so the UI renders correctly.
 */
export function useLibraryFleetSummary(): FleetSummary {
  const [summary] = useState<FleetSummary>(() => ({
    running: 3,
    error: 2,
    listening: 4,
    scheduled: 5,
    idle: 8,
    monthlySpend: 127.45,
  }));
  return summary;
}
