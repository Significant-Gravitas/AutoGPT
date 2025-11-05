import { Button } from "@/components/atoms/Button/Button";
import { PlayIcon } from "lucide-react";
import { useRunGraph } from "./useRunGraph";
import { useGraphStore } from "@/app/(platform)/build/stores/graphStore";
import { useShallow } from "zustand/react/shallow";
import { StopIcon } from "@phosphor-icons/react";
import { cn } from "@/lib/utils";
import { RunInputDialog } from "../RunInputDialog/RunInputDialog";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";

export const RunGraph = () => {
  const {
    handleRunGraph,
    handleStopGraph,
    isSaving,
    openRunInputDialog,
    setOpenRunInputDialog,
  } = useRunGraph();
  const isGraphRunning = useGraphStore(
    useShallow((state) => state.isGraphRunning),
  );

  return (
    <>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="primary"
            size="large"
            className={cn(
              "relative min-w-0 border-none bg-gradient-to-r from-purple-500 to-pink-500 text-lg",
            )}
            onClick={isGraphRunning ? handleStopGraph : handleRunGraph}
          >
            {!isGraphRunning && !isSaving ? (
              <PlayIcon className="size-6" />
            ) : (
              <StopIcon className="size-6" />
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
