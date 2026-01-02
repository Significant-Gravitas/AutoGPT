"use client";

import { OverflowText } from "@/components/atoms/OverflowText/OverflowText";
import { Text } from "@/components/atoms/Text/Text";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { HeartIcon } from "@phosphor-icons/react";
import { InfiniteData } from "@tanstack/react-query";
import Image from "next/image";
import NextLink from "next/link";
import { useEffect, useState } from "react";

import {
  getV2ListFavoriteLibraryAgentsResponse,
  getV2ListLibraryAgentsResponse,
} from "@/app/api/__generated__/endpoints/library/library";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import Avatar, {
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Link } from "@/components/atoms/Link/Link";
import { useToast } from "@/components/molecules/Toast/use-toast";
import BackendAPI, { LibraryAgentID } from "@/lib/autogpt-server-api";
import { cn } from "@/lib/utils";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";

interface LibraryAgentCardProps {
  agent: LibraryAgent;
}

export default function LibraryAgentCard({
  agent: {
    id,
    name,
    description,
    graph_id,
    can_access_graph,
    creator_image_url,
    image_url,
    is_favorite,
  },
}: LibraryAgentCardProps) {
  const isAgentFavoritingEnabled = useGetFlag(Flag.AGENT_FAVORITING);
  const [isFavorite, setIsFavorite] = useState(is_favorite);
  const [isUpdating, setIsUpdating] = useState(false);
  const { toast } = useToast();
  const api = new BackendAPI();
  const queryClient = getQueryClient();

  // Sync local state with prop when it changes (e.g., after query invalidation)
  useEffect(() => {
    setIsFavorite(is_favorite);
  }, [is_favorite]);

  const updateQueryData = (newIsFavorite: boolean) => {
    // Update the agent in all library agent queries
    queryClient.setQueriesData(
      { queryKey: ["/api/library/agents"] },
      (
        oldData:
          | InfiniteData<getV2ListLibraryAgentsResponse, number | undefined>
          | undefined,
      ) => {
        if (!oldData?.pages) return oldData;

        return {
          ...oldData,
          pages: oldData.pages.map((page) => {
            if (page.status !== 200) return page;

            return {
              ...page,
              data: {
                ...page.data,
                agents: page.data.agents.map((agent: LibraryAgent) =>
                  agent.id === id
                    ? { ...agent, is_favorite: newIsFavorite }
                    : agent,
                ),
              },
            };
          }),
        };
      },
    );

    // Update or remove from favorites query based on new state
    queryClient.setQueriesData(
      { queryKey: ["/api/library/agents/favorites"] },
      (
        oldData:
          | InfiniteData<
              getV2ListFavoriteLibraryAgentsResponse,
              number | undefined
            >
          | undefined,
      ) => {
        if (!oldData?.pages) return oldData;

        if (newIsFavorite) {
          // Add to favorites if not already there
          const exists = oldData.pages.some(
            (page) =>
              page.status === 200 &&
              page.data.agents.some((agent: LibraryAgent) => agent.id === id),
          );

          if (!exists) {
            const firstPage = oldData.pages[0];
            if (firstPage?.status === 200) {
              const updatedAgent = {
                id,
                name,
                description,
                graph_id,
                can_access_graph,
                creator_image_url,
                image_url,
                is_favorite: true,
              };

              return {
                ...oldData,
                pages: [
                  {
                    ...firstPage,
                    data: {
                      ...firstPage.data,
                      agents: [updatedAgent, ...firstPage.data.agents],
                      pagination: {
                        ...firstPage.data.pagination,
                        total_items: firstPage.data.pagination.total_items + 1,
                      },
                    },
                  },
                  ...oldData.pages.slice(1).map((page) =>
                    page.status === 200
                      ? {
                          ...page,
                          data: {
                            ...page.data,
                            pagination: {
                              ...page.data.pagination,
                              total_items: page.data.pagination.total_items + 1,
                            },
                          },
                        }
                      : page,
                  ),
                ],
              };
            }
          }
        } else {
          // Remove from favorites
          let removedCount = 0;
          return {
            ...oldData,
            pages: oldData.pages.map((page) => {
              if (page.status !== 200) return page;

              const filteredAgents = page.data.agents.filter(
                (agent: LibraryAgent) => agent.id !== id,
              );

              if (filteredAgents.length < page.data.agents.length) {
                removedCount = 1;
              }

              return {
                ...page,
                data: {
                  ...page.data,
                  agents: filteredAgents,
                  pagination: {
                    ...page.data.pagination,
                    total_items:
                      page.data.pagination.total_items - removedCount,
                  },
                },
              };
            }),
          };
        }

        return oldData;
      },
    );
  };

  const handleToggleFavorite = async (e: React.MouseEvent) => {
    e.preventDefault(); // Prevent navigation when clicking the heart
    e.stopPropagation();

    if (isUpdating || !isAgentFavoritingEnabled) return;

    const newIsFavorite = !isFavorite;

    // Optimistic update
    setIsFavorite(newIsFavorite);
    updateQueryData(newIsFavorite);

    setIsUpdating(true);
    try {
      await api.updateLibraryAgent(id as LibraryAgentID, {
        is_favorite: newIsFavorite,
      });

      toast({
        title: newIsFavorite ? "Added to favorites" : "Removed from favorites",
        description: `${name} has been ${newIsFavorite ? "added to" : "removed from"} your favorites.`,
      });
    } catch (error) {
      // Revert on error
      console.error("Failed to update favorite status:", error);
      setIsFavorite(!newIsFavorite);
      updateQueryData(!newIsFavorite);

      toast({
        title: "Error",
        description: "Failed to update favorite status. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsUpdating(false);
    }
  };

  return (
    <div
      data-testid="library-agent-card"
      data-agent-id={id}
      className="group inline-flex h-[275px] w-full max-w-[434px] flex-col items-start justify-start gap-2.5 rounded-[26px] border border-zinc-100 bg-white transition-all duration-300 hover:shadow-md"
    >
      <NextLink
        href={`/library/agents/${id}`}
        className="relative h-[123px] w-full flex-shrink-0 overflow-hidden rounded-t-[20px] no-underline"
      >
        {!image_url ? (
          <div
            className={`h-full w-full ${
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
            fill
            className="object-cover"
          />
        )}
        {isAgentFavoritingEnabled && (
          <button
            onClick={handleToggleFavorite}
            className={cn(
              "absolute right-4 top-4 rounded-full bg-white/90 p-2 backdrop-blur-sm transition-all duration-200",
              "hover:scale-110 hover:bg-white",
              "focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2",
              isUpdating && "cursor-not-allowed opacity-50",
              !isFavorite && "opacity-0 group-hover:opacity-100",
            )}
            disabled={isUpdating}
            aria-label={
              isFavorite ? "Remove from favorites" : "Add to favorites"
            }
          >
            <HeartIcon
              size={20}
              weight={isFavorite ? "fill" : "regular"}
              className={cn(
                "transition-colors duration-200",
                isFavorite
                  ? "text-red-500"
                  : "text-gray-600 hover:text-red-500",
              )}
            />
          </button>
        )}
        <div className="absolute bottom-4 left-4">
          <Avatar className="h-12 w-12">
            <AvatarImage
              src={
                creator_image_url
                  ? creator_image_url
                  : "/avatar-placeholder.png"
              }
              alt={`${name} creator avatar`}
            />
            <AvatarFallback size={48}>{name.charAt(0)}</AvatarFallback>
          </Avatar>
        </div>
      </NextLink>

      <div className="flex w-full flex-1 flex-col px-4 pb-4 pt-1">
        <Link href={`/library/agents/${id}`}>
          <OverflowText
            value={name}
            variant="body"
            className="mb-4 font-sans text-[1.125rem] tracking-normal"
          />

          <Text
            variant="body"
            className="line-clamp-2 text-ellipsis text-zinc-500"
          >
            {description}
          </Text>
        </Link>

        <div className="flex-grow" />
        {/* Spacer */}

        <div className="items-between mt-4 flex w-full justify-between gap-3">
          <Link href={`/library/agents/${id}`}>See runs</Link>

          {can_access_graph && (
            <Link href={`/build?flowID=${graph_id}`}>Open in builder</Link>
          )}
        </div>
      </div>
    </div>
  );
}
