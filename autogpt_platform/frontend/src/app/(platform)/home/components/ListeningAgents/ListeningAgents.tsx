"use client";

import NextLink from "next/link";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";

interface Props {
  agents: LibraryAgent[];
}

export function ListeningAgents({ agents }: Props) {
  if (agents.length === 0) return null;

  return (
    <section
      aria-label="Listening for triggers"
      className="flex flex-wrap items-center gap-x-1.5 border-t border-zinc-100 px-4 py-2 text-xs text-zinc-500"
    >
      <span
        className="size-1.5 animate-pulse rounded-full bg-purple-500"
        aria-hidden="true"
      />
      <span>Listening for triggers</span>
      <span aria-hidden="true" className="text-zinc-300">
        ·
      </span>
      {agents.map((agent, index) => (
        <span key={agent.id} className="inline-flex items-center">
          <NextLink
            href={`/library/agents/${agent.id}`}
            className="font-medium text-zinc-700 outline-none transition-colors hover:text-zinc-950 focus-visible:underline"
          >
            {agent.name}
          </NextLink>
          {index < agents.length - 1 ? <span aria-hidden="true">,</span> : null}
        </span>
      ))}
    </section>
  );
}
