import { useGraphStore } from "@/app/(platform)/build/stores/graphStore";
import { Button } from "@/components/atoms/Button/Button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { PlayIcon, StopIcon } from "@phosphor-icons/react";
import { useShallow } from "zustand/react/shallow";
import { RunInputDialog } from "../RunInputDialog/RunInputDialog";
import { useRunGraph } from "./useRunGraph";

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

  return (
    <>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            size="icon"
            variant={isGraphRunning ? "destructive" : "primary"}
            onClick={isGraphRunning ? handleStopGraph : handleRunGraph}
            disabled={!flowID || isExecutingGraph || isTerminatingGraph}
            loading={isExecutingGraph || isTerminatingGraph || isSaving}
          >
            {!isGraphRunning ? (
              <PlayIcon className="size-4" />
            ) : (
              <StopIcon className="size-4" />
            )}
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          {isGraphRunning ? "Stop agent" : "Run agent"}
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
