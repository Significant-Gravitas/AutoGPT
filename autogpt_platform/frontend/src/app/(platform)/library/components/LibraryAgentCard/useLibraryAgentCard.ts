"use client";

import { getQueryClient } from "@/lib/react-query/queryClient";
import { useEffect, useState } from "react";

import { usePatchV2UpdateLibraryAgent } from "@/app/api/__generated__/endpoints/library/library";
import { useGetV2GetUserProfile } from "@/app/api/__generated__/endpoints/store/store";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { okData } from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { isLogoutInProgress } from "@/lib/autogpt-server-api/helpers";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { updateFavoriteInQueries } from "./helpers";

interface Props {
  agent: LibraryAgent;
  onFavoriteAdd?: (position: { x: number; y: number }) => void;
}

export function useLibraryAgentCard({ agent, onFavoriteAdd }: Props) {
  const { id, is_favorite, creator_image_url, marketplace_listing } = agent;

  const isFromMarketplace = Boolean(marketplace_listing);
  const [isFavorite, setIsFavorite] = useState(is_favorite);
  const { toast } = useToast();
  const queryClient = getQueryClient();
  const { mutateAsync: updateLibraryAgent } = usePatchV2UpdateLibraryAgent();
  const { user, isLoggedIn } = useSupabase();
  const logoutInProgress = isLogoutInProgress();

  const { data: profile } = useGetV2GetUserProfile({
    query: {
      select: okData,
      enabled: isLoggedIn && !!user && !logoutInProgress,
      queryKey: ["/api/store/profile", user?.id],
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

  async function handleToggleFavorite(
    e: React.MouseEvent,
    position: { x: number; y: number },
  ) {
    e.preventDefault();
    e.stopPropagation();

    const newIsFavorite = !isFavorite;

    // Optimistic update - update UI immediately
    setIsFavorite(newIsFavorite);
    updateQueryData(newIsFavorite);

    // Trigger animation immediately for adding to favorites
    if (newIsFavorite && onFavoriteAdd) {
      onFavoriteAdd(position);
    }

    try {
      await updateLibraryAgent({
        libraryAgentId: id,
        data: { is_favorite: newIsFavorite },
      });
    } catch {
      // Revert on failure
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
