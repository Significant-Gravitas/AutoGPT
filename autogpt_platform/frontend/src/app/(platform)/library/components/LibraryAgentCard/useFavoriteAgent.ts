import { useCallback } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { LibraryAgentID } from "@/lib/autogpt-server-api/types";

export function useFavoriteAgent() {
  const api = useBackendAPI();
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const toggleFavorite = useCallback(
    async (agentId: LibraryAgentID, currentFavoriteStatus: boolean) => {
      const newFavoriteStatus = !currentFavoriteStatus;
      
              // Optimistic update - update the cache immediately for all library agent queries
        queryClient.setQueriesData(
          {
            predicate: (query) => query.queryKey[0] === "v2" && 
                                 query.queryKey[1] === "list" && 
                                 query.queryKey[2] === "library" && 
                                 query.queryKey[3] === "agents",
          },
          (oldData: any) => {
          if (!oldData?.pages) return oldData;
          
          return {
            ...oldData,
            pages: oldData.pages.map((page: any) => ({
              ...page,
              agents: page.agents.map((agent: any) =>
                agent.id === agentId
                  ? { ...agent, is_favorite: newFavoriteStatus }
                  : agent
              ),
            })),
          };
        }
      );

      try {
        await api.updateLibraryAgent(agentId, {
          is_favorite: newFavoriteStatus,
        });
        
        toast({
          title: newFavoriteStatus ? "Added to favorites" : "Removed from favorites",
          description: newFavoriteStatus 
            ? "Agent has been added to your favorites" 
            : "Agent has been removed from your favorites",
          duration: 2000,
        });

        // Invalidate the library agents query to refresh the list with proper sorting
        queryClient.invalidateQueries({
          predicate: (query) => query.queryKey[0] === "v2" && 
                               query.queryKey[1] === "list" && 
                               query.queryKey[2] === "library" && 
                               query.queryKey[3] === "agents",
        });
        
        return newFavoriteStatus;
      } catch (error) {
        // Revert optimistic update on error for all library agent queries
        queryClient.setQueriesData(
          {
            predicate: (query) => query.queryKey[0] === "v2" && 
                                 query.queryKey[1] === "list" && 
                                 query.queryKey[2] === "library" && 
                                 query.queryKey[3] === "agents",
          },
          (oldData: any) => {
            if (!oldData?.pages) return oldData;
            
            return {
              ...oldData,
              pages: oldData.pages.map((page: any) => ({
                ...page,
                agents: page.agents.map((agent: any) =>
                  agent.id === agentId
                    ? { ...agent, is_favorite: currentFavoriteStatus }
                    : agent
                ),
              })),
            };
          }
        );
        
        console.error("Failed to update favorite status:", error);
        toast({
          title: "Error",
          description: "Failed to update favorite status. Please try again.",
          duration: 3000,
          variant: "destructive",
        });
        throw error;
      }
    },
    [api, toast, queryClient]
  );

  return { toggleFavorite };
}