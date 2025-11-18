import { ClockIcon } from "@phosphor-icons/react";
import { RunInputDialog } from "../RunInputDialog/RunInputDialog";
import { useScheduleGraph } from "./useScheduleGraph";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { CronSchedulerDialog } from "../CronSchedulerDialog/CronSchedulerDialog";
import { BuilderActionButton } from "../BuilderActionButton";

export const ScheduleGraph = ({ flowID }: { flowID: string | null }) => {
  const {
    openScheduleInputDialog,
    setOpenScheduleInputDialog,
    handleScheduleGraph,
    openCronSchedulerDialog,
    setOpenCronSchedulerDialog,
  } = useScheduleGraph();
  return (
    <>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <BuilderActionButton
              onClick={handleScheduleGraph}
              disabled={!flowID}
            >
              <ClockIcon className="size-6" />
            </BuilderActionButton>
          </TooltipTrigger>
          <TooltipContent>
            <p>Schedule Graph</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
      <RunInputDialog
        isOpen={openScheduleInputDialog}
        setIsOpen={setOpenScheduleInputDialog}
        purpose="schedule"
      />
      <CronSchedulerDialog
        open={openCronSchedulerDialog}
        setOpen={setOpenCronSchedulerDialog}
        inputs={{}}
        credentials={{}}
      />
    </>
  );
};
