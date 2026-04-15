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
  return false;
}

function hasSensitiveActionBlocks(
  graph: GraphModel | LibraryAgent | Graph,
): boolean {
  if ("has_sensitive_action" in graph) {
    return !!graph.has_sensitive_action;
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
  const showHITLToggle = hasHITLBlocks(graph);
  const showSensitiveActionToggle = hasSensitiveActionBlocks(graph);
  const shouldShowToggle = showHITLToggle || showSensitiveActionToggle;

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

  const [localHITLSafeMode, setLocalHITLSafeMode] = useState<boolean>(true);
  const [localSensitiveActionSafeMode, setLocalSensitiveActionSafeMode] =
    useState<boolean>(false);
  const [isLocalStateLoaded, setIsLocalStateLoaded] = useState<boolean>(false);

  useEffect(() => {
    if (!isAgent && libraryAgent) {
      setLocalHITLSafeMode(
        libraryAgent.settings?.human_in_the_loop_safe_mode ?? true,
      );
      setLocalSensitiveActionSafeMode(
        libraryAgent.settings?.sensitive_action_safe_mode ?? false,
      );
      setIsLocalStateLoaded(true);
    }
  }, [isAgent, libraryAgent]);

  const currentHITLSafeMode = isAgent
    ? (graph.settings?.human_in_the_loop_safe_mode ?? true)
    : localHITLSafeMode;

  const currentSensitiveActionSafeMode = isAgent
    ? (graph.settings?.sensitive_action_safe_mode ?? false)
    : localSensitiveActionSafeMode;

  const isHITLStateUndetermined = isAgent
    ? false
    : isLoading || !isLocalStateLoaded;

  const handleHITLToggle = useCallback(async () => {
    const newSafeMode = !currentHITLSafeMode;

    try {
      await updateGraphSettings({
        graphId,
        data: { human_in_the_loop_safe_mode: newSafeMode },
      });

      if (!isAgent) {
        setLocalHITLSafeMode(newSafeMode);
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
        title: `HITL safe mode ${newSafeMode ? "enabled" : "disabled"}`,
        description: newSafeMode
          ? "Human-in-the-loop blocks will require manual review"
          : "Human-in-the-loop blocks will proceed automatically",
        duration: 2000,
      });
    } catch (error) {
      handleToggleError(error, isAgent, toast);
    }
  }, [
    currentHITLSafeMode,
    graphId,
    isAgent,
    graph.id,
    updateGraphSettings,
    queryClient,
    toast,
  ]);

  const handleSensitiveActionToggle = useCallback(async () => {
    const newSafeMode = !currentSensitiveActionSafeMode;

    try {
      await updateGraphSettings({
        graphId,
        data: { sensitive_action_safe_mode: newSafeMode },
      });

      if (!isAgent) {
        setLocalSensitiveActionSafeMode(newSafeMode);
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
        title: `Sensitive action safe mode ${newSafeMode ? "enabled" : "disabled"}`,
        description: newSafeMode
          ? "Sensitive action blocks will require manual review"
          : "Sensitive action blocks will proceed automatically",
        duration: 2000,
      });
    } catch (error) {
      handleToggleError(error, isAgent, toast);
    }
  }, [
    currentSensitiveActionSafeMode,
    graphId,
    isAgent,
    graph.id,
    updateGraphSettings,
    queryClient,
    toast,
  ]);

  return {
    // HITL safe mode
    currentHITLSafeMode,
    showHITLToggle,
    isHITLStateUndetermined,
    handleHITLToggle,

    // Sensitive action safe mode
    currentSensitiveActionSafeMode,
    showSensitiveActionToggle,
    handleSensitiveActionToggle,

    // General
    isPending,
    shouldShowToggle,

    // Backwards compatibility
    currentSafeMode: currentHITLSafeMode,
    isStateUndetermined: isHITLStateUndetermined,
    handleToggle: handleHITLToggle,
    hasHITLBlocks: showHITLToggle,
  };
}

function handleToggleError(
  error: unknown,
  isAgent: boolean,
  toast: ReturnType<typeof useToast>["toast"],
) {
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
