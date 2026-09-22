"use client";

import { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import { useState } from "react";
import { AICatalogIcon } from "../AICatalogIcon";
import { SectionHeader } from "../SectionHeader";
import { SHELF_GRID, SHELF_PREVIEW_SIZE } from "../Shelf/helpers";
import { ShelfMoreButton } from "../Shelf/ShelfMoreButton";
import { WorkflowTile } from "./components/WorkflowTile";
import { featuredFirst } from "./helpers";

const HEADING_ID = "workflows-heading";

interface Props {
  id: string;
  agents: StoreAgent[];
  featuredAgents: StoreAgent[];
}

export function WorkflowsShelf({ id, agents, featuredAgents }: Props) {
  const [isExpanded, setIsExpanded] = useState(false);
  const ordered = featuredFirst(agents, featuredAgents);

  if (ordered.length === 0) return null;

  const shown = isExpanded ? ordered : ordered.slice(0, SHELF_PREVIEW_SIZE);

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
      <ul className={SHELF_GRID}>
        {shown.map((agent) => (
          <WorkflowTile key={agent.slug} agent={agent} />
        ))}
      </ul>
      {ordered.length > SHELF_PREVIEW_SIZE ? (
        <ShelfMoreButton
          isExpanded={isExpanded}
          count={ordered.length}
          noun="workflows"
          onToggle={() => setIsExpanded(!isExpanded)}
        />
      ) : null}
    </section>
  );
}
