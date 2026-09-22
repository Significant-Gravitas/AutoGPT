import { Button } from "@/components/atoms/Button/Button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { CronSchedulerDialog } from "../CronSchedulerDialog/CronSchedulerDialog";
import { RunInputDialog } from "../RunInputDialog/RunInputDialog";
import { useScheduleGraph } from "./useScheduleGraph";
import { Clock01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
              data-id="schedule-graph-button"
              onClick={handleScheduleGraph}
              disabled={!flowID}
            >
              <Icon icon={Clock01Icon} className="size-4" />
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
