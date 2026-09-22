"use client";

import { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import { Icon } from "@/components/atoms/Icon/Icon";
import { PublishAgentModal } from "@/components/contextual/PublishAgentModal/PublishAgentModal";
import { UserAiIcon } from "@hugeicons/core-free-icons";
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
        titleIcon={<Icon icon={UserAiIcon} size="3.75rem" aria-hidden />}
        title="Workflows"
        titleId={HEADING_ID}
        subtitle="Automations your experts can run — or install one yourself."
      />
      <ul className={SHELF_GRID}>
        {shown.map((agent) => (
          <WorkflowTile key={agent.slug} agent={agent} />
        ))}
      </ul>
      <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
        {ordered.length > SHELF_PREVIEW_SIZE ? (
          <ShelfMoreButton
            isExpanded={isExpanded}
            count={ordered.length}
            noun="workflows"
            onToggle={() => setIsExpanded(!isExpanded)}
          />
        ) : (
          <span />
        )}
        {/* The old full-width creator banner sat at the bottom of the page and
            spoke louder than the shelf it advertised; one line under the
            workflows is the whole invitation. */}
        <PublishAgentModal
          trigger={
            <button
              type="button"
              className="text-sm text-zinc-500 underline-offset-4 transition-colors hover:text-zinc-900 hover:underline"
            >
              Publish your workflows
            </button>
          }
        />
      </div>
    </section>
  );
}
