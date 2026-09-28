"use client";

import { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { PublishAgentModal } from "@/components/contextual/PublishAgentModal/PublishAgentModal";
import { GitCompareArrowsIcon } from "@hugeicons/core-free-icons";
import { useState } from "react";
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
  total: number;
}

export function WorkflowsShelf({ id, agents, featuredAgents, total }: Props) {
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
        titleIcon={
          <Icon icon={GitCompareArrowsIcon} size="2.2rem" aria-hidden />
        }
        title="Workflows"
        titleId={HEADING_ID}
        subtitle="Automations your experts can run — or install one yourself."
      />
      <ul className={SHELF_GRID}>
        {shown.map((agent) => (
          <WorkflowTile key={agent.agent_graph_id} agent={agent} />
        ))}
      </ul>
      <div className="mt-6 flex flex-wrap items-center gap-2">
        {ordered.length > SHELF_PREVIEW_SIZE ? (
          <ShelfMoreButton
            isExpanded={isExpanded}
            count={ordered.length}
            isAll={total <= ordered.length}
            noun="workflows"
            onToggle={() => setIsExpanded(!isExpanded)}
          />
        ) : null}
        {/* The old full-width creator banner sat at the bottom of the page and
            spoke louder than the shelf it advertised; one line under the
            workflows is the whole invitation. */}
        <PublishAgentModal
          trigger={
            <Button variant="secondary" size="small">
              Publish your workflows
            </Button>
          }
        />
      </div>
    </section>
  );
}
