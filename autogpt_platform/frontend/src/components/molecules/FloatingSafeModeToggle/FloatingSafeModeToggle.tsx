import { usePatchV1UpdateGraphSettings } from "@/app/api/__generated__/endpoints/graphs/graphs";
import {
  getGetV2GetLibraryAgentQueryOptions,
  useGetV2GetLibraryAgentByGraphId,
} from "@/app/api/__generated__/endpoints/library/library";
import { GraphModel } from "@/app/api/__generated__/models/graphModel";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { Graph } from "@/lib/autogpt-server-api/types";
import { cn } from "@/lib/utils";
import { ShieldCheckIcon, ShieldIcon } from "@phosphor-icons/react";
import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useState } from "react";
import { useAgentSafeMode } from "@/hooks/useAgentSafeMode";

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
}

export function FloatingSafeModeToggle({
  graph,
  className,
  fullWidth = false,
}: FloatingSafeModeToggleProps) {
  const {
    currentSafeMode,
    isPending,
    shouldShowToggle,
    isStateUndetermined,
    handleToggle,
  } = useAgentSafeMode(graph);

  if (!shouldShowToggle || isStateUndetermined) {
    return null;
  }

  return (
    <div className={cn("fixed z-50", className)}>
      <Tooltip delayDuration={100}>
        <TooltipTrigger asChild>
          <Button
            variant="primary"
            size="small"
            onClick={handleToggle}
            disabled={isPending}
            loading={isPending}
            className={cn(fullWidth ? "w-full" : "")}
          >
            {currentSafeMode! ? (
              <>
                <ShieldCheckIcon weight="bold" size={16} />
                Safe Mode: ON
              </>
            ) : (
              <>
                <ShieldIcon weight="bold" size={16} />
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
