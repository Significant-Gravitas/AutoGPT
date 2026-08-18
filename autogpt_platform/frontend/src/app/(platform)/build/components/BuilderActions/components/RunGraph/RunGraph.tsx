import { useGraphStore } from "@/app/(platform)/build/stores/graphStore";
import { Button } from "@/components/atoms/Button/Button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { useShallow } from "zustand/react/shallow";
import { RunInputDialog } from "../RunInputDialog/RunInputDialog";
import { useRunGraph } from "./useRunGraph";
import { cn } from "@/lib/utils";
import {
  FlaskConicalIcon,
  Loading03Icon,
  PlayIcon,
  StopIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export const RunGraph = ({ flowID }: { flowID: string | null }) => {
  const {
    handleRunGraph,
    handleStopGraph,
    openRunInputDialog,
    setOpenRunInputDialog,
    isExecutingGraph,
    isTerminatingGraph,
    isSaving,
    runTarget,
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
        <Icon icon={Loading03Icon} className={cn(iconClass, "animate-spin")} />
      );
    }

    if (isGraphRunning) {
      return <Icon icon={StopIcon} className={iconClass} />;
    }

    return <Icon icon={PlayIcon} className={iconClass} />;
  };

  return (
    <>
      {/* Simulate button — dry-run, no credentials or credits needed */}
      {!isGraphRunning && (
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              size="icon"
              variant="ghost"
              data-id="simulate-graph-button"
              onClick={() => void handleRunGraph({ dryRun: true })}
              disabled={!flowID || isLoading}
              className="group text-amber-600 hover:bg-amber-50 hover:text-amber-700"
            >
              <Icon
                icon={FlaskConicalIcon}
                className="size-4 transition-transform duration-200 ease-out group-hover:scale-110"
              />
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            Simulate agent (no real execution — LLM-generated outputs)
          </TooltipContent>
        </Tooltip>
      )}

      {/* Run / Stop button */}
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            size="icon"
            variant={isGraphRunning ? "destructive" : "primary"}
            data-id={isGraphRunning ? "stop-graph-button" : "run-graph-button"}
            onClick={
              isGraphRunning ? handleStopGraph : () => void handleRunGraph()
            }
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
        graphID={runTarget?.graphID}
        graphVersion={runTarget?.graphVersion}
      />
    </>
  );
};
