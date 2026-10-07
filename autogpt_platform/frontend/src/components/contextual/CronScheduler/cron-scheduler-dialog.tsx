import { useEffect, useState } from "react";
import { Input } from "@/components/atoms/Input/Input";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { CronScheduler } from "@/components/contextual/CronScheduler/cron-scheduler";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { getTimezoneDisplayName } from "@/lib/timezone-utils";
import { useUserTimezone } from "@/lib/hooks/useUserTimezone";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";

// Base type for cron expression only
type CronOnlyCallback = (cronExpression: string) => void;
// Type for cron expression with schedule name
type CronWithNameCallback = (
  cronExpression: string,
  scheduleName: string,
) => void;

type CronSchedulerDialogProps = {
  open: boolean;
  setOpen: (open: boolean) => void;
  defaultCronExpression?: string;
  title?: string;
} & (
  | {
      // For cases where only cron expression is needed (builder, submission)
      mode: "cron-only";
      onSubmit: CronOnlyCallback;
    }
  | {
      // For cases where schedule name is required (agent run)
      mode: "with-name";
      onSubmit: CronWithNameCallback;
      defaultScheduleName?: string;
    }
);

export function CronSchedulerDialog(props: CronSchedulerDialogProps) {
  const {
    open,
    setOpen,
    defaultCronExpression = "",
    title = "Schedule Task",
  } = props;

  const { toast } = useToast();
  const [cronExpression, setCronExpression] = useState<string>("");
  const [scheduleName, setScheduleName] = useState<string>(
    props.mode === "with-name" ? props.defaultScheduleName || "" : "",
  );

  // Get user's timezone
  const userTimezone = useUserTimezone();
  const timezoneDisplay = getTimezoneDisplayName(userTimezone || "UTC");

  // Reset state when dialog opens
  useEffect(() => {
    if (open) {
      const defaultName =
        props.mode === "with-name" ? props.defaultScheduleName || "" : "";
      setScheduleName(defaultName);
      setCronExpression(defaultCronExpression);
    }
  }, [open, props, defaultCronExpression]);

  const handleDone = () => {
    if (props.mode === "with-name" && !scheduleName.trim()) {
      toast({
        title: "Please enter a schedule name",
        variant: "destructive",
      });
      return;
    }

    // Validate cron expression before proceeding
    if (!cronExpression || cronExpression.trim() === "") {
      toast({
        variant: "destructive",
        title: "Invalid schedule",
        description: "Please enter a valid cron expression",
      });
      return;
    }

    if (props.mode === "with-name") {
      props.onSubmit(cronExpression, scheduleName);
    } else {
      props.onSubmit(cronExpression);
    }
    setOpen(false);
  };

  return (
    <Dialog
      controlled={{ isOpen: open, set: setOpen }}
      title={title}
      styling={{ maxWidth: "600px" }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-4">
          {props.mode === "with-name" && (
            <div className="flex max-w-md flex-col">
              <Input
                id="cron-schedule-name"
                label="Schedule Name"
                labelVariant="body-medium"
                size="small"
                wrapperClassName="mb-0"
                value={scheduleName}
                onChange={(e) => setScheduleName(e.target.value)}
                placeholder="Enter a name for this schedule"
              />
            </div>
          )}

          <CronScheduler
            onCronExpressionChange={setCronExpression}
            initialCronExpression={defaultCronExpression}
            key={`${open}-${defaultCronExpression}`}
          />

          {/* Timezone info */}
          {userTimezone === "not-set" ? (
            <div className="flex items-center gap-2 rounded-md border border-yellow-200 bg-yellow-50 p-3">
              <Icon
                icon={InformationCircleIcon}
                size={16}
                className="text-yellow-600"
              />
              <Text variant="body" className="text-yellow-800">
                No timezone set. Schedule will run in UTC.
                <a href="/settings/account" className="ml-1 underline">
                  Set your timezone
                </a>
              </Text>
            </div>
          ) : (
            <div className="flex items-center gap-2 rounded-md bg-muted/50 p-3">
              <Icon
                icon={InformationCircleIcon}
                size={16}
                className="text-muted-foreground"
              />
              <Text variant="body" tone="muted" unmask={false}>
                Schedule will run in your timezone:{" "}
                <span className="font-medium">{timezoneDisplay}</span>
              </Text>
            </div>
          )}
        </div>
        <div className="mt-8 flex justify-end space-x-2">
          <Button variant="secondary" size="md" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button size="md" onClick={handleDone}>
            Done
          </Button>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}

// Convenience components for common use cases
export function CronExpressionDialog({
  open,
  setOpen,
  onSubmit,
  defaultCronExpression,
  title = "Set Schedule",
}: {
  open: boolean;
  setOpen: (open: boolean) => void;
  onSubmit: (cronExpression: string) => void;
  defaultCronExpression?: string;
  title?: string;
}) {
  return (
    <CronSchedulerDialog
      open={open}
      setOpen={setOpen}
      mode="cron-only"
      onSubmit={onSubmit}
      defaultCronExpression={defaultCronExpression}
      title={title}
    />
  );
}

export function ScheduleTaskDialog({
  open,
  setOpen,
  onSubmit,
  defaultScheduleName,
  defaultCronExpression,
  title = "Schedule Task",
}: {
  open: boolean;
  setOpen: (open: boolean) => void;
  onSubmit: (cronExpression: string, scheduleName: string) => void;
  defaultScheduleName?: string;
  defaultCronExpression?: string;
  title?: string;
}) {
  return (
    <CronSchedulerDialog
      open={open}
      setOpen={setOpen}
      mode="with-name"
      onSubmit={onSubmit}
      defaultScheduleName={defaultScheduleName}
      defaultCronExpression={defaultCronExpression}
      title={title}
    />
  );
}
