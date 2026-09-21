"use client";

import { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import { useState } from "react";
import { AICatalogIcon } from "../AICatalogIcon";
import { AgentsSection } from "../AgentsSection/AgentsSection";
import { SectionHeader } from "../SectionHeader";
import { WorkflowChip } from "./components/WorkflowChip";
import { RAIL_SIZE, featuredFirst } from "./helpers";

const HEADING_ID = "workflows-heading";

interface Props {
  id: string;
  agents: StoreAgent[];
  featuredAgents: StoreAgent[];
}

export function WorkflowsRail({ id, agents, featuredAgents }: Props) {
  const [isExpanded, setIsExpanded] = useState(false);
  const ordered = featuredFirst(agents, featuredAgents);

  if (ordered.length === 0) return null;

  return (
    <section
      id={id}
      aria-labelledby={HEADING_ID}
      className="mb-16 scroll-mt-24"
    >
      <SectionHeader
        size="small"
        titleIcon={<AICatalogIcon size={22} />}
        title="Workflows"
        titleId={HEADING_ID}
        subtitle="Automations your experts can run — or install one yourself."
      />
      {isExpanded ? (
        <AgentsSection agents={ordered} />
      ) : (
        <div className="relative -mx-6 md:-mx-10 lg:-mx-14">
          <ul className="flex snap-x gap-3 overflow-x-auto px-6 pb-2 scrollbar-none md:px-10 lg:px-14">
            {ordered.slice(0, RAIL_SIZE).map((agent) => (
              <WorkflowChip key={agent.slug} agent={agent} />
            ))}
          </ul>
          <div className="pointer-events-none absolute inset-y-0 right-0 w-16 bg-gradient-to-l from-[rgb(246,247,248)] to-transparent" />
        </div>
      )}
      {ordered.length > 1 ? (
        <button
          type="button"
          aria-expanded={isExpanded}
          onClick={() => setIsExpanded(!isExpanded)}
          className="mt-4 text-sm font-medium text-zinc-500 transition-colors hover:text-zinc-900"
        >
          {isExpanded
            ? "Show fewer"
            : `Show all ${ordered.length.toLocaleString()} workflows`}
        </button>
      ) : null}
    </section>
  );
}
