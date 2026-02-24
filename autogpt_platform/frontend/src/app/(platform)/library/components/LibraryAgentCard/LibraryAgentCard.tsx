"use client";

import { Text } from "@/components/atoms/Text/Text";
import { CaretCircleRightIcon } from "@phosphor-icons/react";
import Image from "next/image";
import NextLink from "next/link";
import { motion } from "framer-motion";

import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import Avatar, {
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Link } from "@/components/atoms/Link/Link";
import { AgentCardMenu } from "./components/AgentCardMenu";
import { FavoriteButton } from "./components/FavoriteButton";
import { useLibraryAgentCard } from "./useLibraryAgentCard";
import { useFavoriteAnimation } from "../../context/FavoriteAnimationContext";

interface Props {
  agent: LibraryAgent;
  draggable?: boolean;
}

export function LibraryAgentCard({ agent, draggable = true }: Props) {
  const { id, name, graph_id, can_access_graph, image_url } = agent;
  const { triggerFavoriteAnimation } = useFavoriteAnimation();

  function handleDragStart(e: React.DragEvent<HTMLDivElement>) {
    e.dataTransfer.setData("application/agent-id", id);
    e.dataTransfer.effectAllowed = "move";
  }

  const {
    isFromMarketplace,
    isFavorite,
    profile,
    creator_image_url,
    handleToggleFavorite,
  } = useLibraryAgentCard({
    agent,
    onFavoriteAdd: triggerFavoriteAnimation,
  });

  return (
    <div
      draggable={draggable}
      onDragStart={handleDragStart}
      className="[@media(pointer:fine)]:cursor-grab [@media(pointer:fine)]:active:cursor-grabbing"
    >
      <motion.div
        layoutId={`agent-card-${id}`}
        data-testid="library-agent-card"
        data-agent-id={id}
        className="group relative inline-flex h-[10.625rem] w-full max-w-[25rem] flex-col items-start justify-start gap-2.5 rounded-medium border border-zinc-100 bg-white hover:shadow-md"
        transition={{
          type: "spring",
          damping: 25,
          stiffness: 300,
        }}
        style={{ willChange: "transform" }}
      >
        <NextLink href={`/library/agents/${id}`} className="flex-shrink-0">
          <div className="relative flex items-center gap-2 px-4 pt-3">
            <Avatar className="h-4 w-4 rounded-full">
              <AvatarImage
                src={
                  isFromMarketplace
                    ? creator_image_url || "/avatar-placeholder.png"
                    : profile?.avatar_url || "/avatar-placeholder.png"
                }
                alt={`${name} creator avatar`}
              />
              <AvatarFallback size={48}>{name.charAt(0)}</AvatarFallback>
            </Avatar>
            <Text
              variant="small-medium"
              className="uppercase tracking-wide text-zinc-400"
            >
              {isFromMarketplace ? "FROM MARKETPLACE" : "Built by you"}
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
          <Link
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
          </Link>

          <div className="mt-auto flex w-full justify-start gap-6 border-t border-zinc-100 pb-1 pt-3">
            <Link
              href={`/library/agents/${id}`}
              data-testid="library-agent-card-see-runs-link"
              className="flex items-center gap-1 text-[13px]"
            >
              See runs <CaretCircleRightIcon size={20} />
            </Link>

            {can_access_graph && (
              <Link
                href={`/build?flowID=${graph_id}`}
                data-testid="library-agent-card-open-in-builder-link"
                className="flex items-center gap-1 text-[13px]"
                isExternal
              >
                Open in builder <CaretCircleRightIcon size={20} />
              </Link>
            )}
          </div>
        </div>
      </motion.div>
    </div>
  );
}
