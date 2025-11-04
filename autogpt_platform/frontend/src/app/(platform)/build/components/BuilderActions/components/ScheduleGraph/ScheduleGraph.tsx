import { Button } from "@/components/atoms/Button/Button";
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

export const ScheduleGraph = () => {
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
              variant="primary"
              size="large"
              className={"relative min-w-0 border-none text-lg"}
              onClick={handleScheduleGraph}
            >
              <ClockIcon className="size-6" />
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
