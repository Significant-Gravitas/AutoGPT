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
    const iconClass = cn("size-4", !isLoading && "group-hover:scale-110");

    if (isLoading) {
      return <Icon icon={Loading03Icon} spin className={iconClass} />;
    }

    if (isGraphRunning) {
      return <Icon icon={StopIcon} motion="transform" className={iconClass} />;
    }

    return <Icon icon={PlayIcon} motion="transform" className={iconClass} />;
  };

  return (
    <>
      {/* Simulate button — dry-run, no credentials or credits needed */}
      {!isGraphRunning && (
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              size="icon"
              variant="ghost-warning"
              data-id="simulate-graph-button"
              onClick={() => void handleRunGraph({ dryRun: true })}
              disabled={!flowID || isLoading}
              className="group"
            >
              <Icon
                icon={FlaskConicalIcon}
                motion="transform"
                className="size-4 group-hover:scale-110"
              />
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            Simulate workflow (no real execution — LLM-generated outputs)
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
              ? "Stop workflow"
              : "Run workflow"}
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
