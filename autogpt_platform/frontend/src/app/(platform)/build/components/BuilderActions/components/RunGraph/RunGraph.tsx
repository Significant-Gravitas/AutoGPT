import { useRunGraph } from "./useRunGraph";
import { useGraphStore } from "@/app/(platform)/build/stores/graphStore";
import { useShallow } from "zustand/react/shallow";
import { PlayIcon, StopIcon } from "@phosphor-icons/react";
import { cn } from "@/lib/utils";
import { RunInputDialog } from "../RunInputDialog/RunInputDialog";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { BuilderActionButton } from "../BuilderActionButton";

export const RunGraph = ({ flowID }: { flowID: string | null }) => {
  const {
    handleRunGraph,
    handleStopGraph,
    openRunInputDialog,
    setOpenRunInputDialog,
    isExecutingGraph,
    isSaving,
  } = useRunGraph();
  const isGraphRunning = useGraphStore(
    useShallow((state) => state.isGraphRunning),
  );

  return (
    <>
      <Tooltip>
        <TooltipTrigger asChild>
          <BuilderActionButton
            className={cn(
              isGraphRunning &&
                "border-red-500 bg-gradient-to-br from-red-400 to-red-500 shadow-[inset_0_2px_0_0_rgba(255,255,255,0.5),0_2px_4px_0_rgba(0,0,0,0.2)]",
            )}
            onClick={isGraphRunning ? handleStopGraph : handleRunGraph}
            disabled={!flowID || isExecutingGraph}
            isLoading={isExecutingGraph || isSaving}
          >
            {!isGraphRunning ? (
              <PlayIcon className="size-6 drop-shadow-sm" />
            ) : (
              <StopIcon className="size-6 drop-shadow-sm" />
            )}
          </BuilderActionButton>
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
