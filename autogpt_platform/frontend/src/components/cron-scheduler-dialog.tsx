import { useEffect, useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { Separator } from "@/components/ui/separator";
import { CronScheduler } from "@/components/cron-scheduler";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";

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
