import { useGraphStore } from "@/app/(platform)/build/stores/graphStore";
import { Button } from "@/components/atoms/Button/Button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { CircleNotchIcon, PlayIcon, StopIcon } from "@phosphor-icons/react";
import { useShallow } from "zustand/react/shallow";
import { RunInputDialog } from "../RunInputDialog/RunInputDialog";
import { useRunGraph } from "./useRunGraph";
import { cn } from "@/lib/utils";

export const RunGraph = ({ flowID }: { flowID: string | null }) => {
  const {
    handleRunGraph,
    handleStopGraph,
    openRunInputDialog,
    setOpenRunInputDialog,
    isExecutingGraph,
    isTerminatingGraph,
    isSaving,
  } = useRunGraph();
  const isGraphRunning = useGraphStore(
    useShallow((state) => state.isGraphRunning),
  );

  const isLoading = isExecutingGraph || isTerminatingGraph || isSaving;

  // Determine which icon to show with proper animation
  const renderIcon = () => {
    const iconClass = cn(
      "size-4 transition-transform duration-200 ease-out",
      !isLoading && "group-hover:scale-110",
    );

    if (isLoading) {
      return (
        <CircleNotchIcon
          className={cn(iconClass, "animate-spin")}
          weight="bold"
        />
      );
    }

    if (isGraphRunning) {
      return <StopIcon className={iconClass} weight="fill" />;
    }

    return <PlayIcon className={iconClass} weight="fill" />;
  };

  return (
    <>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            size="icon"
            variant={isGraphRunning ? "destructive" : "primary"}
            data-id={isGraphRunning ? "stop-graph-button" : "run-graph-button"}
            onClick={isGraphRunning ? handleStopGraph : handleRunGraph}
            disabled={!flowID || isLoading}
            className="group"
          >
            {renderIcon()}
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          {isLoading
            ? "Processing..."
            : isGraphRunning
              ? "Stop agent"
              : "Run agent"}
        </TooltipContent>
      </Tooltip>
      <RunInputDialog
        isOpen={openRunInputDialog}
        setIsOpen={setOpenRunInputDialog}
        purpose="run"
      />
    </>
  );
};
