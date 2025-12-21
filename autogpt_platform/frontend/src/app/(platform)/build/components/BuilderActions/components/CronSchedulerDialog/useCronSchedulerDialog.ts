import { useGetV1GetUserTimezone } from "@/app/api/__generated__/endpoints/auth/auth";
import { usePostV1CreateExecutionSchedule } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { getTimezoneDisplayName } from "@/lib/timezone-utils";
import { parseAsInteger, parseAsString, useQueryStates } from "nuqs";
import { useEffect, useState } from "react";

export const useCronSchedulerDialog = ({
  open,
  setOpen,
  inputs,
  credentials,
  defaultCronExpression = "",
}: {
  open: boolean;
  setOpen: (open: boolean) => void;
  inputs: Record<string, any>;
  credentials: Record<string, any>;
  defaultCronExpression?: string;
}) => {
  const { toast } = useToast();
  const [cronExpression, setCronExpression] = useState<string>("");
  const [scheduleName, setScheduleName] = useState<string>("");

  const [{ flowID, flowVersion }] = useQueryStates({
    flowID: parseAsString,
    flowVersion: parseAsInteger,
    flowExecutionID: parseAsString,
  });

  const { data: userTimezone } = useGetV1GetUserTimezone({
    query: {
      select: (res) => (res.status === 200 ? res.data.timezone : undefined),
    },
  });
  const timezoneDisplay = getTimezoneDisplayName(userTimezone || "UTC");

  const { mutateAsync: createSchedule, isPending: isCreatingSchedule } =
    usePostV1CreateExecutionSchedule({
      mutation: {
        onSuccess: (response) => {
          if (response.status === 200) {
            setOpen(false);
            toast({
              title: "Schedule created",
              description: "Schedule created successfully",
            });
          }
        },
        onError: (error) => {
          toast({
            variant: "destructive",
            title: "Failed to create schedule",
            description:
              (error.detail as string) ?? "An unexpected error occurred.",
          });
        },
      },
    });

  useEffect(() => {
    if (open) {
      setCronExpression(defaultCronExpression);
    }
  }, [open, defaultCronExpression]);

  const handleCreateSchedule = async () => {
    if (!cronExpression || cronExpression.trim() === "") {
      toast({
        variant: "destructive",
        title: "Invalid schedule",
        description: "Please enter a valid cron expression",
      });
      return;
    }

    await createSchedule({
      graphId: flowID || "",
      data: {
        name: scheduleName,
        graph_version: flowID ? flowVersion : undefined,
        cron: cronExpression,
        inputs: inputs,
        credentials: credentials,
      },
    });
    setOpen(false);
  };

  return {
    cronExpression,
    setCronExpression,
    userTimezone,
    timezoneDisplay,
    handleCreateSchedule,
    setScheduleName,
    scheduleName,
    isCreatingSchedule,
  };
};
