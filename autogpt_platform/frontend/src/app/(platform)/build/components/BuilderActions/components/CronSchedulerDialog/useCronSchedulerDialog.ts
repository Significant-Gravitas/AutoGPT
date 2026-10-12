import type { ScheduleCreationRequest } from "@/app/api/__generated__/models/scheduleCreationRequest";
import { usePostV1CreateExecutionSchedule } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { trackScheduleCreatedGoal } from "@/services/analytics/activation-goals";
import { useUserTimezone } from "@/lib/hooks/useUserTimezone";
import { getTimezoneDisplayName } from "@/lib/timezone-utils";
import {
  invalidateAllScheduleQueries,
  reconcileAfterScheduleWriteError,
} from "@/services/schedules/invalidate-schedules";
import { resendUnconfirmedWriteOnce } from "@/services/schedules/scheduler-availability";
import { createSubmissionKeys } from "@/services/schedules/submission-keys";
import { useQueryClient } from "@tanstack/react-query";
import { parseAsInteger, parseAsString, useQueryStates } from "nuqs";
import { useEffect, useState } from "react";

export function useCronSchedulerDialog({
  open,
  setOpen,
  inputs,
  credentials,
  defaultCronExpression = "",
}: {
  open: boolean;
  setOpen: (open: boolean) => void;
  inputs: ScheduleCreationRequest["inputs"];
  credentials: ScheduleCreationRequest["credentials"];
  defaultCronExpression?: string;
}) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [cronExpression, setCronExpression] = useState<string>("");
  const [scheduleName, setScheduleName] = useState<string>("");
  const [scheduleNameError, setScheduleNameError] = useState<string>("");
  const [submissionKeys] = useState(createSubmissionKeys);

  const [{ flowID, flowVersion }] = useQueryStates({
    flowID: parseAsString,
    flowVersion: parseAsInteger,
    flowExecutionID: parseAsString,
  });

  const userTimezone = useUserTimezone();
  const timezoneDisplay = getTimezoneDisplayName(userTimezone || "UTC");

  const { mutateAsync: createSchedule, isPending: isCreatingSchedule } =
    usePostV1CreateExecutionSchedule({
      mutation: {
        ...resendUnconfirmedWriteOnce,
        onSuccess: (response) => {
          if (response.status === 200) {
            setOpen(false);
            toast({
              title: "Schedule created",
              description: "Schedule created successfully",
            });
            if (flowID) trackScheduleCreatedGoal({ id: flowID }, "builder");
            invalidateAllScheduleQueries(queryClient, flowID ?? undefined);
            submissionKeys.reset();
          }
        },
        onError: (error) => {
          // The scheduler may have created it without confirming; show it if so.
          // A retry reuses the submission's key, so it can't add a second one.
          reconcileAfterScheduleWriteError(
            queryClient,
            error,
            flowID ?? undefined,
          );
          toast({
            variant: "destructive",
            title: "Failed to create schedule",
            description:
              error instanceof Error
                ? error.message
                : "An unexpected error occurred.",
          });
        },
      },
    });

  useEffect(() => {
    if (open) {
      setCronExpression(defaultCronExpression);
    }
  }, [open, defaultCronExpression]);

  useEffect(() => {
    if (open) {
      setScheduleName("");
      setScheduleNameError("");
    }
  }, [open]);

  function handleScheduleNameChange(name: string) {
    setScheduleName(name);
    if (name.trim() !== "") {
      setScheduleNameError("");
    }
  }

  async function handleCreateSchedule() {
    if (!scheduleName || scheduleName.trim() === "") {
      setScheduleNameError("Schedule name is required");
      toast({
        variant: "destructive",
        title: "Invalid schedule",
        description: "Please enter a schedule name",
      });
      return;
    }

    if (!cronExpression || cronExpression.trim() === "") {
      toast({
        variant: "destructive",
        title: "Invalid schedule",
        description: "Please enter a valid cron expression",
      });
      return;
    }

    const data = {
      name: scheduleName.trim(),
      graph_version: flowID ? flowVersion : undefined,
      cron: cronExpression,
      inputs: inputs,
      credentials: credentials,
    };
    await createSchedule({
      graphId: flowID || "",
      data: {
        ...data,
        schedule_id: submissionKeys.keyFor({ graphId: flowID, ...data }),
      },
    });
  }

  return {
    cronExpression,
    setCronExpression,
    userTimezone,
    timezoneDisplay,
    handleCreateSchedule,
    setScheduleName: handleScheduleNameChange,
    scheduleName,
    scheduleNameError,
    isCreatingSchedule,
  };
}
