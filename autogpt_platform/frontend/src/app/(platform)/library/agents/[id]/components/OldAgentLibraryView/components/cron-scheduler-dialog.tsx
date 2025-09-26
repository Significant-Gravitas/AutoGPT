import { useEffect, useState } from "react";
import { Input } from "@/components/__legacy__/ui/input";
import { Button } from "@/components/__legacy__/ui/button";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { CronScheduler } from "@/app/(platform)/library/agents/[id]/components/OldAgentLibraryView/components/cron-scheduler";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useGetV1GetUserTimezone } from "@/app/api/__generated__/endpoints/auth/auth";
import { getTimezoneDisplayName } from "@/lib/timezone-utils";
import { InfoIcon } from "lucide-react";
import { useOnboarding } from "@/providers/onboarding/onboarding-provider";

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
  const { completeStep } = useOnboarding();

  // Get user's timezone
  const { data: userTimezone } = useGetV1GetUserTimezone({
    query: {
      select: (res) => (res.status === 200 ? res.data.timezone : undefined),
    },
  });
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
    completeStep("SCHEDULE_AGENT");
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
            <div className="flex max-w-[448px] flex-col space-y-2">
              <label className="text-sm font-medium">Schedule Name</label>
              <Input
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
            <div className="flex items-center gap-2 rounded-md border border-amber-200 bg-amber-50 p-3">
              <InfoIcon className="h-4 w-4 text-amber-600" />
              <p className="text-sm text-amber-800">
                No timezone set. Schedule will run in UTC.
                <a href="/profile/settings" className="ml-1 underline">
                  Set your timezone
                </a>
              </p>
            </div>
          ) : (
            <div className="flex items-center gap-2 rounded-md bg-muted/50 p-3">
              <InfoIcon className="h-4 w-4 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">
                Schedule will run in your timezone:{" "}
                <span className="font-medium">{timezoneDisplay}</span>
              </p>
            </div>
          )}
        </div>
        <div className="mt-8 flex justify-end space-x-2">
          <Button variant="outline" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button onClick={handleDone}>Done</Button>
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
