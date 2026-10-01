"use client";

import { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import { isLocalStoreMediaUrl } from "@/lib/store-media";
import Image from "next/image";
import { useState } from "react";
import { ShelfTile } from "../../Shelf/ShelfTile";

interface Props {
  agent: StoreAgent;
}

export function WorkflowTile({ agent }: Props) {
  const [imageError, setImageError] = useState(false);
  const runs = agent.runs ?? 0;

  return (
    <ShelfTile
      testId="workflow-tile"
      href={`/marketplace/agent/${encodeURIComponent(agent.creator)}/${encodeURIComponent(agent.slug)}`}
      mediaClassName="bg-violet-100 ring-1 ring-black/5"
      media={
        agent.agent_image && !imageError ? (
          <Image
            src={agent.agent_image}
            unoptimized={isLocalStoreMediaUrl(agent.agent_image)}
            alt=""
            fill
            sizes="56px"
            className="object-cover"
            onError={() => setImageError(true)}
          />
        ) : null
      }
      title={agent.agent_name}
      subtitle={runs === 0 ? "No runs" : `${runs.toLocaleString()} runs`}
    />
  );
}
