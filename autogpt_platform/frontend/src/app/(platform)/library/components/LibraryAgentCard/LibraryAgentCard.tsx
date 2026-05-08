"use client";

import { Text } from "@/components/atoms/Text/Text";
import { EyeIcon, ChatCircleDotsIcon } from "@phosphor-icons/react";
import Image from "next/image";
import NextLink from "next/link";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";

import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { cn } from "@/lib/utils";
import { AgentCardMenu } from "./components/AgentCardMenu";
import { FavoriteButton } from "./components/FavoriteButton";
import { useLibraryAgentCard } from "./useLibraryAgentCard";
import { useFavoriteAnimation } from "../../context/FavoriteAnimationContext";
import { StatusBadge } from "../StatusBadge/StatusBadge";
import { ContextualActionButton } from "../ContextualActionButton/ContextualActionButton";
import type { AgentStatusInfo } from "../../types";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface Props {
  agent: LibraryAgent;
  statusInfo: AgentStatusInfo;
  draggable?: boolean;
}

export function LibraryAgentCard({
  agent,
  statusInfo,
  draggable = true,
}: Props) {
  const { id, name, image_url } = agent;
  const router = useRouter();
  const { triggerFavoriteAnimation } = useFavoriteAnimation();

  function handleDragStart(e: React.DragEvent<HTMLDivElement>) {
    e.dataTransfer.setData("application/agent-id", id);
    e.dataTransfer.effectAllowed = "move";
  }

  const { isFavorite, handleToggleFavorite } = useLibraryAgentCard({
    agent,
    onFavoriteAdd: triggerFavoriteAnimation,
  });

  const hasError = statusInfo.status === "error";

  const card = (
    <div
      draggable={draggable}
      onDragStart={handleDragStart}
      className="[@media(pointer:fine)]:cursor-grab [@media(pointer:fine)]:active:cursor-grabbing"
    >
      <motion.div
        layoutId={`agent-card-${id}`}
        data-testid="library-agent-card"
        data-agent-id={id}
        className={cn(
          "group relative inline-flex h-auto min-h-[10.625rem] w-full max-w-[25rem] flex-col items-start justify-start gap-2.5 rounded-medium border bg-white hover:shadow-md",
          hasError ? "border-red-400" : "border-zinc-100",
        )}
        transition={{
          type: "spring",
          damping: 25,
          stiffness: 300,
        }}
        style={{ willChange: "transform" }}
      >
        <NextLink href={`/library/agents/${id}`} className="flex-shrink-0">
          <div className="relative flex items-center gap-3 pl-2 pr-4 pt-3">
            <StatusBadge status={statusInfo.status} />
            <Text variant="small" className="text-zinc-400">
              {statusInfo.totalRuns} tasks
            </Text>
          </div>
        </NextLink>
        <FavoriteButton
          isFavorite={isFavorite}
          onClick={handleToggleFavorite}
          className="absolute right-10 top-0"
        />
        <AgentCardMenu agent={agent} />

        <div className="flex w-full flex-1 flex-col px-4 pb-2">
          <NextLink
            href={`/library/agents/${id}`}
            className="flex w-full items-start justify-between gap-2 no-underline hover:no-underline focus:ring-0"
          >
            <Text
              variant="h5"
              data-testid="library-agent-card-name"
              className="line-clamp-3 hyphens-auto break-words no-underline hover:no-underline"
            >
              {name}
            </Text>

            {!image_url ? (
              <div
                className={`h-[3.64rem] w-[6.70rem] flex-shrink-0 rounded-small ${
                  [
                    "bg-gradient-to-r from-green-200 to-blue-200",
                    "bg-gradient-to-r from-pink-200 to-purple-200",
                    "bg-gradient-to-r from-yellow-200 to-orange-200",
                    "bg-gradient-to-r from-blue-200 to-cyan-200",
                    "bg-gradient-to-r from-indigo-200 to-purple-200",
                  ][parseInt(id.slice(0, 8), 16) % 5]
                }`}
                style={{
                  backgroundSize: "200% 200%",
                  animation: "gradient 15s ease infinite",
                }}
              />
            ) : (
              <Image
                src={image_url}
                alt={`${name} preview image`}
                width={107}
                height={58}
                className="flex-shrink-0 rounded-small object-cover"
              />
            )}
          </NextLink>

          <div className="mt-4 flex w-full items-center justify-end gap-1 border-t border-zinc-100 pb-0 pt-2">
            <button
              type="button"
              onClick={() => router.push(`/library/agents/${id}`)}
              data-testid="library-agent-card-see-runs-link"
              className="inline-flex items-center gap-1 rounded-md px-2 py-1.5 text-[13px] font-medium text-zinc-600 transition-colors hover:bg-zinc-50 hover:text-zinc-800"
            >
              <EyeIcon size={14} className="shrink-0" />
              See tasks
            </button>
            <ContextualActionButton
              status={statusInfo.status}
              agentID={id}
              executionID={statusInfo.activeExecutionID ?? undefined}
            />
            <button
              type="button"
              onClick={() => {
                const prompt = encodeURIComponent(
                  `Tell me about ${name}, its current status, recent runs and how can I get the most out of it`,
                );
                router.push(`/copilot?autosubmit=true#prompt=${prompt}`);
              }}
              className="inline-flex items-center gap-1 rounded-md px-2 py-1.5 text-[13px] font-medium text-zinc-600 transition-colors hover:bg-zinc-50 hover:text-zinc-800"
            >
              <ChatCircleDotsIcon size={14} className="shrink-0" />
              Chat
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );

  if (hasError && statusInfo.lastError) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>{card}</TooltipTrigger>
        <TooltipContent className="max-w-xs text-red-600">
          {statusInfo.lastError}
        </TooltipContent>
      </Tooltip>
    );
  }

  return card;
}
