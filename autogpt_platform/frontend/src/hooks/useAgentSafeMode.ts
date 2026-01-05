import { useCallback, useState, useEffect } from "react";
import { usePatchV1UpdateGraphSettings } from "@/app/api/__generated__/endpoints/graphs/graphs";
import {
  getGetV2GetLibraryAgentQueryOptions,
  useGetV2GetLibraryAgentByGraphId,
} from "@/app/api/__generated__/endpoints/library/library";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { GraphModel } from "@/app/api/__generated__/models/graphModel";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { okData } from "@/app/api/helpers";
import { useQueryClient } from "@tanstack/react-query";
import { Graph } from "@/lib/autogpt-server-api/types";

function getGraphId(graph: GraphModel | LibraryAgent | Graph): string {
  if ("graph_id" in graph) return graph.graph_id || "";
  return (graph.id || "").toString();
}

function hasHITLBlocks(graph: GraphModel | LibraryAgent | Graph): boolean {
  if ("has_human_in_the_loop" in graph) {
    return !!graph.has_human_in_the_loop;
  }

  if (isLibraryAgent(graph)) {
    return graph.settings?.human_in_the_loop_safe_mode !== null;
  }

  return false;
}

function isLibraryAgent(
  graph: GraphModel | LibraryAgent | Graph,
): graph is LibraryAgent {
  return "graph_id" in graph && "settings" in graph;
}

export function useAgentSafeMode(graph: GraphModel | LibraryAgent | Graph) {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const graphId = getGraphId(graph);
  const isAgent = isLibraryAgent(graph);
  const shouldShowToggle = hasHITLBlocks(graph);

  const { mutateAsync: updateGraphSettings, isPending } =
    usePatchV1UpdateGraphSettings();

  const { data: libraryAgent, isLoading } = useGetV2GetLibraryAgentByGraphId(
    graphId,
    {},
    {
      query: {
        enabled: !isAgent && shouldShowToggle,
        select: okData,
      },
    },
  );

  const [localSafeMode, setLocalSafeMode] = useState<boolean | null>(null);

  useEffect(() => {
    if (!isAgent && libraryAgent) {
      const backendValue = libraryAgent.settings?.human_in_the_loop_safe_mode;
      if (backendValue !== undefined) {
        setLocalSafeMode(backendValue);
      }
    }
  }, [isAgent, libraryAgent]);

  const currentSafeMode = isAgent
    ? graph.settings?.human_in_the_loop_safe_mode
    : localSafeMode;

  const isStateUndetermined = isAgent
    ? graph.settings?.human_in_the_loop_safe_mode == null
    : isLoading || localSafeMode === null;

  const handleToggle = useCallback(async () => {
    const newSafeMode = !currentSafeMode;

    try {
      await updateGraphSettings({
        graphId,
        data: { human_in_the_loop_safe_mode: newSafeMode },
      });

      if (!isAgent) {
        setLocalSafeMode(newSafeMode);
      }

      if (isAgent) {
        queryClient.invalidateQueries({
          queryKey: getGetV2GetLibraryAgentQueryOptions(graph.id.toString())
            .queryKey,
        });
      }

      queryClient.invalidateQueries({
        queryKey: ["v1", "graphs", graphId, "executions"],
      });
      queryClient.invalidateQueries({ queryKey: ["v2", "executions"] });

      toast({
        title: `Safe mode ${newSafeMode ? "enabled" : "disabled"}`,
        description: newSafeMode
          ? "Human-in-the-loop blocks will require manual review"
          : "Human-in-the-loop blocks will proceed automatically",
        duration: 2000,
      });
    } catch (error) {
      const isNotFoundError =
        error instanceof Error &&
        (error.message.includes("404") || error.message.includes("not found"));

      if (!isAgent && isNotFoundError) {
        toast({
          title: "Safe mode not available",
          description:
            "To configure safe mode, please save this graph to your library first.",
          variant: "destructive",
        });
      } else {
        toast({
          title: "Failed to update safe mode",
          description:
            error instanceof Error
              ? error.message
              : "An unexpected error occurred.",
          variant: "destructive",
        });
      }
    }
  }, [
    currentSafeMode,
    graphId,
    isAgent,
    graph.id,
    updateGraphSettings,
    queryClient,
    toast,
  ]);

  return {
    currentSafeMode,
    isPending,
    shouldShowToggle,
    isStateUndetermined,
    handleToggle,
    hasHITLBlocks: shouldShowToggle,
  };
}
