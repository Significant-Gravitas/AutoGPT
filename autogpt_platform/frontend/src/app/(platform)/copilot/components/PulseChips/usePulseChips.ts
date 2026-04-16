"use client";

import { useLibraryAgents } from "@/hooks/useLibraryAgents/useLibraryAgents";
import { useSitrepItems } from "@/app/(platform)/library/components/SitrepItem/useSitrepItems";
import type { PulseChipData } from "./types";
import { useMemo } from "react";

// TODO: remove QA fakes before merging
const QA_FAKES: PulseChipData[] = [
  {
    id: "qa-1",
    agentID: "fake-1",
    name: "SEO Blog Writer with Advanced Keyword Research and Content Optimization",
    status: "running",
    priority: "running",
    shortMessage:
      "Writing a comprehensive long-form article on the latest AI trends in enterprise software development and deployment",
  },
  {
    id: "qa-2",
    agentID: "fake-2",
    name: "Multi-Cloud Data Pipeline Monitor and Alerting System",
    status: "error",
    priority: "error",
    shortMessage:
      "Connection to the primary data warehouse timed out after 30 retries — fallback region also unreachable",
  },
  {
    id: "qa-3",
    agentID: "fake-3",
    name: "Social Media Cross-Platform Scheduler and Analytics Dashboard",
    status: "idle",
    priority: "success",
    shortMessage:
      "All 12 scheduled posts across Twitter, LinkedIn, and Instagram were published successfully with engagement tracking enabled",
  },
  {
    id: "qa-4",
    agentID: "fake-4",
    name: "Customer Support Triage and Automatic Escalation Handler",
    status: "running",
    priority: "stale",
    shortMessage:
      "3 high-priority tickets awaiting classification — SLA breach warning for 2 enterprise accounts pending review",
  },
];

export function usePulseChips(): PulseChipData[] {
  const { agents } = useLibraryAgents();

  const sitrepItems = useSitrepItems(agents, 5);

  return useMemo(() => {
    const real = sitrepItems.map((item) => ({
      id: item.id,
      agentID: item.agentID,
      name: item.agentName,
      status: item.status,
      priority: item.priority,
      shortMessage: item.message,
    }));
    return [...real, ...QA_FAKES];
  }, [sitrepItems]);
}
