"use client";

import { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import Avatar, {
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { cn } from "@/lib/utils";
import Image from "next/image";
import { useRef, useState } from "react";
import { AddToLibraryButton } from "../AddToLibraryButton/AddToLibraryButton";

interface Props {
  agent: StoreAgent;
  backgroundColor: string;
}

// Soft top wash per featured slot — same treatment as the expert cards so
// the two families read as one system. Body stays white.
const WASHES: Record<string, string> = {
  violet:
    "bg-[radial-gradient(120%_100%_at_50%_0%,rgba(139,92,246,0.10),transparent_70%)]",
  blue: "bg-[radial-gradient(120%_100%_at_50%_0%,rgba(59,130,246,0.10),transparent_70%)]",
  green:
    "bg-[radial-gradient(120%_100%_at_50%_0%,rgba(34,197,94,0.10),transparent_70%)]",
};

function getWash(bg: string) {
  if (bg.includes("violet")) return WASHES.violet;
  if (bg.includes("blue")) return WASHES.blue;
  if (bg.includes("green")) return WASHES.green;
  return WASHES.violet;
}

export function FeaturedAgentCard({ agent, backgroundColor }: Props) {
  const [imageError, setImageError] = useState(false);
  const [imageLoaded, setImageLoaded] = useState(false);
  const titleRef = useRef<HTMLSpanElement>(null);
  const [isTitleTruncated, setIsTitleTruncated] = useState(false);

  function checkTitleOverflow() {
    const el = titleRef.current;
    if (el) setIsTitleTruncated(el.scrollHeight > el.clientHeight);
  }

  return (
    <div
      className="group relative flex h-full w-full max-w-md cursor-pointer flex-col items-start overflow-hidden rounded-2xl border border-zinc-200/80 bg-white p-5 shadow-[0_1px_2px_rgba(16,24,40,0.04)] transition-all duration-200 hover:-translate-y-0.5 hover:border-zinc-300 hover:shadow-[0_16px_40px_-16px_rgba(16,24,40,0.18)]"
      data-testid="featured-store-card"
    >
      <div
        className={cn(
          "pointer-events-none absolute inset-x-0 top-0 h-32 opacity-60 transition-opacity duration-200 group-hover:opacity-100",
          getWash(backgroundColor),
        )}
      />
      <div className="relative aspect-[2/1.2] w-full overflow-hidden rounded-xl ring-1 ring-black/5 md:aspect-[2.17/1]">
        {agent.agent_image && !imageError ? (
          <>
            {!imageLoaded && (
              <Skeleton className="absolute inset-0 rounded-xl" />
            )}
            <Image
              src={agent.agent_image}
              alt={`${agent.agent_name} preview image`}
              fill
              className="object-cover"
              onLoad={() => setImageLoaded(true)}
              onError={() => setImageError(true)}
            />
          </>
        ) : (
          <div className="absolute inset-0 rounded-xl bg-violet-100" />
        )}
      </div>

      <div className="relative mt-4 flex w-full flex-1 flex-col">
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <span
                ref={titleRef}
                onPointerEnter={checkTitleOverflow}
                className="line-clamp-1 block min-w-0 font-sans text-lg font-semibold tracking-[-0.01em] text-zinc-900"
              >
                {agent.agent_name}
              </span>
            </TooltipTrigger>
            {isTitleTruncated && (
              <TooltipContent>
                <p>{agent.agent_name}</p>
              </TooltipContent>
            )}
          </Tooltip>
        </TooltipProvider>
        {agent.creator && (
          <div className="mt-1.5 flex items-center gap-1.5">
            <Avatar className="h-5 w-5 shrink-0">
              {agent.creator_avatar && (
                <AvatarImage
                  src={agent.creator_avatar}
                  alt={`${agent.creator} creator avatar`}
                />
              )}
              <AvatarFallback size={20}>
                {agent.creator.charAt(0)}
              </AvatarFallback>
            </Avatar>
            <span className="truncate text-[13px] text-zinc-500">
              by {agent.creator}
            </span>
          </div>
        )}
        <p className="mt-3 line-clamp-3 text-sm leading-relaxed text-zinc-600">
          {agent.description}
        </p>
      </div>

      <div className="relative mt-auto flex w-full items-center pt-3">
        <span className="text-xs text-zinc-500">
          {agent.runs === 0
            ? "No runs"
            : `${(agent.runs ?? 0).toLocaleString()} runs`}
        </span>
      </div>
      {agent.creator && agent.slug && agent.agent_graph_id && (
        <div className="absolute bottom-4 right-4">
          <AddToLibraryButton
            creatorSlug={agent.creator}
            agentSlug={agent.slug}
            agentName={agent.agent_name}
            agentGraphID={agent.agent_graph_id}
          />
        </div>
      )}
    </div>
  );
}
