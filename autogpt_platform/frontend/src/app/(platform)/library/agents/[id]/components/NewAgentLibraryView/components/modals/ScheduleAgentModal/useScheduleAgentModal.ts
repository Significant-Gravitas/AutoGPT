import { usePostV1CreateExecutionSchedule as useCreateSchedule } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { trackScheduleCreatedGoal } from "@/services/analytics/activation-goals";
import {
  invalidateAllScheduleQueries,
  reconcileAfterScheduleWriteError,
} from "@/services/schedules/invalidate-schedules";
import { resendUnconfirmedWriteOnce } from "@/services/schedules/scheduler-availability";
import { createSubmissionKeys } from "@/services/schedules/submission-keys";
import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useMemo, useState } from "react";

interface UseScheduleAgentModalCallbacks {
  onCreateSchedule?: (schedule: GraphExecutionJobInfo) => void;
  onClose?: () => void;
}

export function useScheduleAgentModal(
  agent: LibraryAgent,
  inputValues: Record<string, any>,
  inputCredentials: Record<string, any>,
  callbacks?: UseScheduleAgentModalCallbacks,
) {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const defaultScheduleName = useMemo(() => `Run ${agent.name}`, [agent.name]);
  const [scheduleName, setScheduleName] = useState(defaultScheduleName);
  const [cronExpression, setCronExpression] = useState(
    agent.recommended_schedule_cron || "0 9 * * 1",
  );
  const [submissionKeys] = useState(createSubmissionKeys);

  const createScheduleMutation = useCreateSchedule({
    mutation: {
      ...resendUnconfirmedWriteOnce,
      onSuccess: (response) => {
        if (response.status === 200) {
          toast({
            title: "Schedule created",
          });
          trackScheduleCreatedGoal(
            { id: agent.graph_id, name: agent.name },
            "library",
          );
          callbacks?.onCreateSchedule?.(response.data);
          invalidateAllScheduleQueries(queryClient, agent.graph_id);
          submissionKeys.reset();
          // Reset form
          setScheduleName(defaultScheduleName);
          setCronExpression(agent.recommended_schedule_cron || "0 9 * * 1");
          callbacks?.onClose?.();
        }
      },
      onError: (error: any) => {
        // The scheduler may have created it without confirming; show it if so.
        // A retry reuses the submission's key, so it can't add a second one.
        reconcileAfterScheduleWriteError(queryClient, error, agent.graph_id);
        toast({
          title: "❌ Failed to create schedule",
          description: error.message || "An unexpected error occurred.",
          variant: "destructive",
        });
      },
    },
  });

  const handleSchedule = useCallback(
    (scheduleName: string, cronExpression: string) => {
      if (!scheduleName.trim()) {
        toast({
          title: "⚠️ Schedule name required",
          description: "Please provide a name for your schedule.",
          variant: "destructive",
        });
        return Promise.reject(new Error("Schedule name required"));
      }

      const data = {
        name: scheduleName,
        cron: cronExpression,
        inputs: inputValues,
        graph_version: agent.graph_version,
        credentials: inputCredentials,
      };
      return new Promise<void>((resolve, reject) => {
        createScheduleMutation.mutate(
          {
            graphId: agent.graph_id,
            data: {
              ...data,
              schedule_id: submissionKeys.keyFor({
                graphId: agent.graph_id,
                ...data,
              }),
            },
          },
          {
            onSuccess: () => resolve(),
            onError: (error) => reject(error),
          },
        );
      });
    },
    [
      agent.graph_id,
      agent.graph_version,
      inputValues,
      inputCredentials,
      createScheduleMutation,
      submissionKeys,
      toast,
    ],
  );

  const handleSetScheduleName = useCallback((name: string) => {
    setScheduleName(name);
  }, []);

  const handleSetCronExpression = useCallback((expression: string) => {
    setCronExpression(expression);
  }, []);

  const resetForm = useCallback(() => {
    setScheduleName(defaultScheduleName);
    setCronExpression(agent.recommended_schedule_cron || "0 9 * * 1");
  }, [defaultScheduleName, agent.recommended_schedule_cron]);

  return {
    // State
    scheduleName,
    cronExpression,

    // Loading state
    isCreatingSchedule: createScheduleMutation.isPending,

    // Actions
    handleSchedule,
    handleSetScheduleName,
    handleSetCronExpression,
    resetForm,
  };
}
