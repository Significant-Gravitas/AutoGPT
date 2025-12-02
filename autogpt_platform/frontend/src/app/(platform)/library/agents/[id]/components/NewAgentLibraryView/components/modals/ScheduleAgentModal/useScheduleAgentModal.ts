import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useState, useCallback, useMemo } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useToast } from "@/components/molecules/Toast/use-toast";
import {
  usePostV1CreateExecutionSchedule as useCreateSchedule,
  getGetV1ListExecutionSchedulesForAGraphQueryKey,
} from "@/app/api/__generated__/endpoints/schedules/schedules";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";

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

  const createScheduleMutation = useCreateSchedule({
    mutation: {
      onSuccess: (response) => {
        if (response.status === 200) {
          toast({
            title: "Schedule created",
          });
          callbacks?.onCreateSchedule?.(response.data);
          // Invalidate schedules list for this graph
          queryClient.invalidateQueries({
            queryKey: getGetV1ListExecutionSchedulesForAGraphQueryKey(
              agent.graph_id,
            ),
          });
          // Reset form
          setScheduleName(defaultScheduleName);
          setCronExpression(agent.recommended_schedule_cron || "0 9 * * 1");
          callbacks?.onClose?.();
        }
      },
      onError: (error: any) => {
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

      return new Promise<void>((resolve, reject) => {
        createScheduleMutation.mutate(
          {
            graphId: agent.graph_id,
            data: {
              name: scheduleName,
              cron: cronExpression,
              inputs: inputValues,
              graph_version: agent.graph_version,
              credentials: inputCredentials,
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
