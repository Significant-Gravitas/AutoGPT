import { Button } from "@/components/atoms/Button/Button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { ClockIcon } from "@phosphor-icons/react";
import { CronSchedulerDialog } from "../CronSchedulerDialog/CronSchedulerDialog";
import { RunInputDialog } from "../RunInputDialog/RunInputDialog";
import { useScheduleGraph } from "./useScheduleGraph";

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
            <Button
              variant="outline"
              size="icon"
              onClick={handleScheduleGraph}
              disabled={!flowID}
            >
              <ClockIcon className="size-4" />
            </Button>
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
