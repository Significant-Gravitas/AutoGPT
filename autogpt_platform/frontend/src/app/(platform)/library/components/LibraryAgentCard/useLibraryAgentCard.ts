"use client";

import { getQueryClient } from "@/lib/react-query/queryClient";
import { useEffect, useState } from "react";

import { usePatchV2UpdateLibraryAgent } from "@/app/api/__generated__/endpoints/library/library";
import { useGetV2GetUserProfile } from "@/app/api/__generated__/endpoints/store/store";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { okData } from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { updateFavoriteInQueries } from "./helpers";

interface Props {
  agent: LibraryAgent;
}

export function useLibraryAgentCard({ agent }: Props) {
  const { id, name, is_favorite, creator_image_url, marketplace_listing } =
    agent;

  const isFromMarketplace = Boolean(marketplace_listing);
  const [isFavorite, setIsFavorite] = useState(is_favorite);
  const { toast } = useToast();
  const queryClient = getQueryClient();
  const { mutateAsync: updateLibraryAgent } = usePatchV2UpdateLibraryAgent();

  const { data: profile } = useGetV2GetUserProfile({
    query: {
      select: okData,
    },
  });

  useEffect(() => {
    setIsFavorite(is_favorite);
  }, [is_favorite]);

  function updateQueryData(newIsFavorite: boolean) {
    updateFavoriteInQueries({
      queryClient,
      agentId: id,
      agent,
      newIsFavorite,
    });
  }

  async function handleToggleFavorite(e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();

    const newIsFavorite = !isFavorite;

    setIsFavorite(newIsFavorite);
    updateQueryData(newIsFavorite);

    try {
      await updateLibraryAgent({
        libraryAgentId: id,
        data: { is_favorite: newIsFavorite },
      });

      toast({
        title: newIsFavorite ? "Added to favorites" : "Removed from favorites",
        description: `${name} has been ${newIsFavorite ? "added to" : "removed from"} your favorites.`,
      });
    } catch {
      setIsFavorite(!newIsFavorite);
      updateQueryData(!newIsFavorite);

      toast({
        title: "Error",
        description: "Failed to update favorite status. Please try again.",
        variant: "destructive",
      });
    }
  }

  return {
    isFromMarketplace,
    isFavorite,
    profile,
    creator_image_url,
    handleToggleFavorite,
  };
}
