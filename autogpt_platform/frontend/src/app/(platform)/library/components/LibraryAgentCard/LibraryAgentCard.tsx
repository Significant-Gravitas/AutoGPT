"use client";

import { Text } from "@/components/atoms/Text/Text";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { CaretCircleRightIcon, HeartIcon } from "@phosphor-icons/react";
import { InfiniteData } from "@tanstack/react-query";
import Image from "next/image";
import NextLink from "next/link";
import { useEffect, useState } from "react";

import {
  getV2ListFavoriteLibraryAgentsResponse,
  getV2ListLibraryAgentsResponse,
} from "@/app/api/__generated__/endpoints/library/library";
import { useGetV2GetUserProfile } from "@/app/api/__generated__/endpoints/store/store";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { okData } from "@/app/api/helpers";
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
    marketplace_listing,
  },
}: LibraryAgentCardProps) {
  const isFromMarketplace = Boolean(marketplace_listing);
  const isAgentFavoritingEnabled = useGetFlag(Flag.AGENT_FAVORITING);
  const [isFavorite, setIsFavorite] = useState(is_favorite);
  const [isUpdating, setIsUpdating] = useState(false);
  const { toast } = useToast();
  const api = new BackendAPI();
  const queryClient = getQueryClient();

  const { data: profile } = useGetV2GetUserProfile({
    query: {
      select: okData,
    },
  });

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
      className="group inline-flex h-[10.625rem] w-full max-w-[25rem] flex-col items-start justify-start gap-2.5 rounded-medium border border-zinc-100 bg-white transition-all duration-300 hover:shadow-md"
    >
      <NextLink
        href={`/library/agents/${id}`}
        className="overflow-hidde w-full flex-shrink-0"
      >
        <div className="flex items-center gap-2 px-4 pt-3">
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
        {isAgentFavoritingEnabled && (
          <button
            onClick={handleToggleFavorite}
            className={cn(
              "rounded-full bg-white/90 p-2 backdrop-blur-sm transition-all duration-200",
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
      </NextLink>

      <div className="flex w-full flex-1 flex-col px-4 pb-2">
        <Link
          href={`/library/agents/${id}`}
          className="flex w-full items-start justify-between gap-2 no-underline hover:no-underline"
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
              fill
              className="h-[3.64rem] w-[6.70rem] flex-shrink-0 rounded-small object-cover"
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
    </div>
  );
}
