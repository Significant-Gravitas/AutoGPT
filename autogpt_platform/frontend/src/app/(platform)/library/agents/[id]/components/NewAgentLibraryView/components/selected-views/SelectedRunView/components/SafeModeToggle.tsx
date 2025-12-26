import { GraphModel } from "@/app/api/__generated__/models/graphModel";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { Graph } from "@/lib/autogpt-server-api/types";
import { cn } from "@/lib/utils";
import { ShieldCheckIcon, ShieldIcon } from "@phosphor-icons/react";
import { useAgentSafeMode } from "@/hooks/useAgentSafeMode";

interface Props {
  graph: GraphModel | LibraryAgent | Graph;
  className?: string;
  fullWidth?: boolean;
}

export function SafeModeToggle({ graph }: Props) {
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
    <Button
      variant="icon"
      key={graph.id}
      size="icon"
      aria-label={
        currentSafeMode!
          ? "Safe Mode: ON. Human in the loop blocks require manual review"
          : "Safe Mode: OFF. Human in the loop blocks proceed automatically"
      }
      onClick={handleToggle}
      className={cn(isPending ? "opacity-0" : "opacity-100")}
    >
      {currentSafeMode! ? (
        <>
          <ShieldCheckIcon weight="bold" size={16} />
        </>
      ) : (
        <>
          <ShieldIcon weight="bold" size={16} />
        </>
      )}
    </Button>
  );
}
