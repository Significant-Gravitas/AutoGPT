import { Button } from "@/components/atoms/Button/Button";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { InfoIcon } from "lucide-react";
import { CronScheduler } from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/ScheduleAgentModal/components/CronScheduler/CronScheduler";
import { Text } from "@/components/atoms/Text/Text";
import { useCronSchedulerDialog } from "./useCronSchedulerDialog";
import { Input } from "@/components/atoms/Input/Input";

type CronSchedulerDialogProps = {
  open: boolean;
  setOpen: (open: boolean) => void;
  inputs: Record<string, any>;
  credentials: Record<string, any>;
  defaultCronExpression?: string;
  title?: string;
};

export function CronSchedulerDialog({
  open,
  setOpen,

  defaultCronExpression = "",
  title = "Schedule Graph",
  inputs,
  credentials,
}: CronSchedulerDialogProps) {
  const {
    setCronExpression,
    userTimezone,
    timezoneDisplay,
    handleCreateSchedule,
    scheduleName,
    setScheduleName,
    isCreatingSchedule,
  } = useCronSchedulerDialog({
    open,
    setOpen,
    inputs,
    credentials,
    defaultCronExpression,
  });
  return (
    <Dialog
      controlled={{ isOpen: open, set: setOpen }}
      title={title}
      styling={{ maxWidth: "600px", minWidth: "600px" }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-4">
          <Input
            id="schedule-name"
            label="Schedule Name"
            placeholder="Enter schedule name"
            size="small"
            className="max-w-80"
            value={scheduleName}
            onChange={(e) => setScheduleName(e.target.value)}
          />

          <CronScheduler
            onCronExpressionChange={setCronExpression}
            initialCronExpression={defaultCronExpression}
            key={`${open}-${defaultCronExpression}`}
          />

          {/* Timezone info */}
          {userTimezone === "not-set" ? (
            <div className="flex items-center gap-2 rounded-xlarge border border-amber-200 bg-amber-50 p-3">
              <InfoIcon className="h-4 w-4 text-amber-600" />
              <Text variant="body" className="text-amber-800">
                No timezone set. Schedule will run in UTC.
                <a href="/profile/settings" className="ml-1 underline">
                  Set your timezone
                </a>
              </Text>
            </div>
          ) : (
            <div className="flex items-center gap-2 rounded-xlarge bg-muted/50 p-3">
              <InfoIcon className="h-4 w-4 text-muted-foreground" />
              <Text variant="body">
                Schedule will run in your timezone:{" "}
                <Text variant="body-medium" as="span">
                  {timezoneDisplay}
                </Text>
              </Text>
            </div>
          )}
        </div>
        <div className="mt-8 flex justify-end space-x-2">
          <Button
            variant="outline"
            onClick={() => setOpen(false)}
            className="h-fit"
          >
            Cancel
          </Button>
          <Button
            loading={isCreatingSchedule}
            disabled={isCreatingSchedule}
            onClick={handleCreateSchedule}
            className="h-fit"
          >
            {isCreatingSchedule ? "Creating schedule..." : "Done"}
          </Button>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
