import { useCallback, useState, useEffect } from "react";
import { ShieldIcon, ShieldCheckIcon } from "@phosphor-icons/react";
import { Button } from "@/components/atoms/Button/Button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { usePatchV1UpdateGraphSettings } from "@/app/api/__generated__/endpoints/graphs/graphs";
import {
  getGetV2GetLibraryAgentQueryOptions,
  useGetV2GetLibraryAgentByGraphId,
} from "@/app/api/__generated__/endpoints/library/library";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { cn } from "@/lib/utils";
import { GraphModel } from "@/app/api/__generated__/models/graphModel";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
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

interface FloatingSafeModeToggleProps {
  graph: GraphModel | LibraryAgent | Graph;
  className?: string;
  fullWidth?: boolean;
  variant?: "white" | "black";
}

export function FloatingSafeModeToggle({
  graph,
  className,
  fullWidth = false,
  variant = "white",
}: FloatingSafeModeToggleProps) {
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
    { query: { enabled: !isAgent && shouldShowToggle } },
  );

  const [localSafeMode, setLocalSafeMode] = useState<boolean | null>(null);

  useEffect(() => {
    if (!isAgent && libraryAgent?.status === 200) {
      const backendValue =
        libraryAgent.data?.settings?.human_in_the_loop_safe_mode;
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

  if (!shouldShowToggle || isStateUndetermined) {
    return null;
  }

  return (
    <div className={cn(variant === "black" ? "fixed z-50" : "", className)}>
      <Tooltip delayDuration={100}>
        <TooltipTrigger asChild>
          <Button
            variant="secondary"
            size="small"
            onClick={handleToggle}
            disabled={isPending}
            loading={isPending}
            className={cn(
              fullWidth ? "w-full" : "",
              variant === "black"
                ? "bg-gray-800 text-white hover:bg-gray-700"
                : "",
            )}
          >
            {currentSafeMode! ? (
              <>
                <ShieldCheckIcon className="h-4 w-4" />
                Safe Mode: ON
              </>
            ) : (
              <>
                <ShieldIcon className="h-4 w-4" />
                Safe Mode: OFF
              </>
            )}
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          <div className="text-center">
            <div className="font-medium">
              Safe Mode: {currentSafeMode! ? "ON" : "OFF"}
            </div>
            <div className="mt-1 text-xs text-muted-foreground">
              {currentSafeMode!
                ? "HITL blocks require manual review"
                : "HITL blocks proceed automatically"}
            </div>
          </div>
        </TooltipContent>
      </Tooltip>
    </div>
  );
}
