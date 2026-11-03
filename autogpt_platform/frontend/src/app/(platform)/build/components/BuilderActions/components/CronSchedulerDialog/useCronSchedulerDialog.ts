import type { ScheduleCreationRequest } from "@/app/api/__generated__/models/scheduleCreationRequest";
import { usePostV1CreateExecutionSchedule } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useUserTimezone } from "@/lib/hooks/useUserTimezone";
import { getTimezoneDisplayName } from "@/lib/timezone-utils";
import { invalidateAllScheduleQueries } from "@/services/schedules/invalidate-schedules";
import { useQueryClient } from "@tanstack/react-query";
import { parseAsInteger, parseAsString, useQueryStates } from "nuqs";
import { useEffect, useState } from "react";
import { useCreateTeamSelection } from "@/components/contextual/TeamPicker/useCreateTeamSelection";
import {
  CreateSurface,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";
import { useBuilderTenantScope } from "@/app/(platform)/build/hooks/useBuilderTenantScope";

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
  const tenantScope = useBuilderTenantScope();
  const [scheduleNameError, setScheduleNameError] = useState<string>("");

  const [{ flowID, flowVersion }] = useQueryStates({
    flowID: parseAsString,
    flowVersion: parseAsInteger,
    flowExecutionID: parseAsString,
  });

  const userTimezone = useUserTimezone();
  const timezoneDisplay = getTimezoneDisplayName(userTimezone || "UTC");
  const { teamId, setTeamId, hasTeams, teamRequestInit, isReady } =
    useCreateTeamSelection(CreateSurface.BuilderSchedule);

  const { mutateAsync: createSchedule, isPending: isCreatingSchedule } =
    usePostV1CreateExecutionSchedule({
      request: flowID
        ? getTenantRequestInit(
            tenantScope.organizationId,
            tenantScope.teamId,
            tenantScope.isReady,
          )
        : teamRequestInit,
      mutation: {
        onSuccess: (response) => {
          if (response.status === 200) {
            setOpen(false);
            toast({
              title: "Schedule created",
              description: "Schedule created successfully",
            });
            invalidateAllScheduleQueries(queryClient, flowID ?? undefined);
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
    if (!(flowID ? tenantScope.isReady : isReady)) return;
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

    await createSchedule({
      graphId: flowID || "",
      data: {
        name: scheduleName.trim(),
        graph_version: flowID ? flowVersion : undefined,
        cron: cronExpression,
        inputs: inputs,
        credentials: credentials,
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
    teamId,
    setTeamId,
    hasTeams: !flowID && hasTeams,
    isReady: flowID ? tenantScope.isReady : isReady,
  };
}
