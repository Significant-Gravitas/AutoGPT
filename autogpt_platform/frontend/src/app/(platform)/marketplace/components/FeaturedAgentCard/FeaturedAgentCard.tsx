"use client";

import { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import Avatar, {
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Text } from "@/components/atoms/Text/Text";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import Image from "next/image";
import { useRef, useState } from "react";
import { AddToLibraryButton } from "../AddToLibraryButton/AddToLibraryButton";

interface Props {
  agent: StoreAgent;
  backgroundColor: string;
}

function getAccentTextClass(bg: string) {
  if (bg.includes("violet")) return "text-violet-500 hover:text-violet-800";
  if (bg.includes("blue")) return "text-blue-500 hover:text-blue-800";
  if (bg.includes("green")) return "text-green-500 hover:text-green-800";
  return "text-zinc-500 hover:text-zinc-800";
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
      className={`relative flex h-[28rem] w-full max-w-md cursor-pointer flex-col items-start rounded-2xl p-4 shadow-md transition-all duration-300 hover:shadow-lg ${backgroundColor} border`}
      data-testid="featured-store-card"
    >
      {/* Image */}
      <div className="relative aspect-[2/1.2] w-full overflow-hidden rounded-xl md:aspect-[2.17/1]">
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
          <div className="absolute inset-0 rounded-xl bg-violet-50" />
        )}
      </div>

      <div className="mt-3 flex w-full flex-1 flex-col">
        {/* Agent Name and Creator */}
        <div className="flex w-full flex-col">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span
                  ref={titleRef}
                  onPointerEnter={checkTitleOverflow}
                  className="line-clamp-2 block min-h-[2lh] min-w-0 leading-tight"
                >
                  <Text variant="h4" as="span" className="leading-tight">
                    {agent.agent_name}
                  </Text>
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
            <div className="mt-3 flex items-center gap-2">
              <Avatar className="h-6 w-6 shrink-0">
                {agent.creator_avatar && (
                  <AvatarImage
                    src={agent.creator_avatar}
                    alt={`${agent.creator} creator avatar`}
                  />
                )}
                <AvatarFallback size={32}>
                  {agent.creator.charAt(0)}
                </AvatarFallback>
              </Avatar>
              <Text variant="body-medium" className="truncate">
                by {agent.creator}
              </Text>
            </div>
          )}
        </div>

        {/* Description */}
        <div className="mt-2.5 flex w-full flex-col">
          <Text variant="body" className="line-clamp-3 leading-normal">
            {agent.description}
          </Text>
        </div>
      </div>

      {/* Stats */}
      <Text variant="body" className="absolute bottom-4 left-4 text-zinc-500">
        {agent.runs === 0
          ? "No runs"
          : `${(agent.runs ?? 0).toLocaleString()} runs`}
      </Text>
      {agent.creator && agent.slug && agent.agent_graph_id && (
        <div className="absolute bottom-2 right-0">
          <AddToLibraryButton
            creatorSlug={agent.creator}
            agentSlug={agent.slug}
            agentName={agent.agent_name}
            agentGraphID={agent.agent_graph_id}
            className={getAccentTextClass(backgroundColor)}
          />
        </div>
      )}
    </div>
  );
}
