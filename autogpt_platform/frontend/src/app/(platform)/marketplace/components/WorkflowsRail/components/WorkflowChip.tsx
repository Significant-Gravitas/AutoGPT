"use client";

import { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import { isLocalStoreMediaUrl } from "@/lib/store-media";
import Image from "next/image";
import Link from "next/link";
import { useState } from "react";

interface Props {
  agent: StoreAgent;
}

export function WorkflowChip({ agent }: Props) {
  const [imageError, setImageError] = useState(false);

  return (
    <li className="shrink-0 snap-start">
      <Link
        href={`/marketplace/agent/${encodeURIComponent(agent.creator)}/${encodeURIComponent(agent.slug)}`}
        data-testid="workflow-chip"
        className="flex w-64 items-center gap-3 rounded-xl border border-zinc-200/80 bg-white p-2.5 outline-none transition-colors hover:border-zinc-300 focus-visible:ring-2 focus-visible:ring-violet-600"
      >
        <span className="relative h-14 w-14 shrink-0 overflow-hidden rounded-lg bg-violet-100 ring-1 ring-black/5">
          {agent.agent_image && !imageError ? (
            <Image
              src={agent.agent_image}
              unoptimized={isLocalStoreMediaUrl(agent.agent_image)}
              alt=""
              fill
              sizes="56px"
              className="object-cover"
              onError={() => setImageError(true)}
            />
          ) : null}
        </span>
        <span className="min-w-0">
          <span
            title={agent.agent_name}
            className="block truncate text-sm font-medium text-zinc-900"
          >
            {agent.agent_name}
          </span>
          <span className="block truncate text-[13px] text-zinc-500">
            {agent.runs === 0
              ? "No runs"
              : `${agent.runs.toLocaleString()} runs`}
          </span>
        </span>
      </Link>
    </li>
  );
}
