import { useEffect, useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { Separator } from "@/components/ui/separator";
import { CronScheduler } from "@/components/cron-scheduler";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { useGetV1GetUserTimezone } from "@/app/api/__generated__/endpoints/auth/auth";
import { getTimezoneDisplayName } from "@/lib/timezone-utils";
import { InfoIcon } from "lucide-react";

type CronSchedulerDialogProps = {
  open: boolean;
  setOpen: (open: boolean) => void;
  afterCronCreation: (cronExpression: string, scheduleName: string) => void;
  defaultScheduleName?: string;
};

export function CronSchedulerDialog({
  open,
  setOpen,
  afterCronCreation,
  defaultScheduleName = "",
}: CronSchedulerDialogProps) {
  const { toast } = useToast();
  const [cronExpression, setCronExpression] = useState<string>("");
  const [scheduleName, setScheduleName] = useState<string>(defaultScheduleName);

  // Get user's timezone
  const { data: timezoneData } = useGetV1GetUserTimezone();
  const userTimezone = timezoneData?.data?.timezone || "UTC";
  const timezoneDisplay = getTimezoneDisplayName(userTimezone);

  // Reset state when dialog opens
  useEffect(() => {
    if (open) {
      setScheduleName(defaultScheduleName);
      setCronExpression("");
    }
  }, [open]);

  const handleDone = () => {
    if (!scheduleName.trim()) {
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

    afterCronCreation(cronExpression, scheduleName);
    setOpen(false);
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent>
        <DialogTitle>Schedule Task</DialogTitle>
        <div className="p-2">
          <div className="flex flex-col gap-4">
            <div className="flex flex-col space-y-2">
              <label className="text-sm font-medium">Schedule Name</label>
              <Input
                value={scheduleName}
                onChange={(e) => setScheduleName(e.target.value)}
                placeholder="Enter a name for this schedule"
              />
            </div>

            <CronScheduler onCronExpressionChange={setCronExpression} />

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

          <Separator className="my-4" />

          <div className="flex justify-end space-x-2">
            <Button variant="outline" onClick={() => setOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleDone}>Done</Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
