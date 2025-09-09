"use client";

import React from "react";
import { useFavoriteAgents } from "../../hooks/useFavoriteAgents";
import LibraryAgentCard from "../LibraryAgentCard/LibraryAgentCard";
import { useGetFlag } from "@/services/feature-flags/use-get-flag";
import { Heart } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

export default function FavoritesSection() {
  const isAgentFavoritingEnabled = useGetFlag("isAgentFavoritingEnabled");
  const { data, isLoading, error } = useFavoriteAgents();

  // Only show this section if the feature flag is enabled
  if (!isAgentFavoritingEnabled) {
    return null;
  }

  // Don't show the section if there's an error
  if (error) {
    return null;
  }

  // Get all favorite agents from all pages
  const favoriteAgents = data?.pages.flatMap((page) => page.agents) || [];

  // Don't show the section if there are no favorites
  if (!isLoading && favoriteAgents.length === 0) {
    return null;
  }

  return (
    <div className="mb-8">
      <div className="mb-4 flex items-center gap-2">
        <Heart className="h-5 w-5 fill-red-500 text-red-500" />
        <h2 className="text-lg font-semibold">Favorites</h2>
        {!isLoading && (
          <span className="text-sm text-muted-foreground">
            ({favoriteAgents.length})
          </span>
        )}
      </div>
      
      <div className="relative">
        {isLoading ? (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {[...Array(4)].map((_, i) => (
              <Skeleton key={i} className="h-48 w-full rounded-lg" />
            ))}
          </div>
        ) : (
          <div className={cn(
            "grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4",
            favoriteAgents.length > 4 && "overflow-x-auto"
          )}>
            {favoriteAgents.map((agent) => (
              <LibraryAgentCard
                key={agent.id}
                {...agent}
                className="min-w-[280px]"
              />
            ))}
          </div>
        )}
      </div>
      
      {favoriteAgents.length > 0 && (
        <div className="mt-6 border-t pt-6" />
      )}
    </div>
  );
}