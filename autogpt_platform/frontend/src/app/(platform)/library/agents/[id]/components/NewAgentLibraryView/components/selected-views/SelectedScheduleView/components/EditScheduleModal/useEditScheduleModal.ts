"use client";

import { getGetV1ListGraphExecutionsQueryKey } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { usePatchV1UpdateExecutionSchedule } from "@/app/api/__generated__/endpoints/schedules/schedules";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { invalidateAllScheduleQueries } from "@/services/schedules/invalidate-schedules";
import { useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import {
  parseCronToForm,
  validateSchedule,
} from "../../../../modals/ScheduleAgentModal/components/ModalScheduleSection/helpers";

export function useEditScheduleModal(
  graphId: string,
  schedule: GraphExecutionJobInfo,
) {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [isOpen, setIsOpen] = useState(false);
  const [name, setName] = useState(schedule.name);

  const parsed = useMemo(() => parseCronToForm(schedule.cron), [schedule.cron]);
  const [repeat, setRepeat] = useState<string>(parsed?.repeat || "daily");
  const [selectedDays, setSelectedDays] = useState<string[]>(
    parsed?.selectedDays || [],
  );
  const [time, setTime] = useState<string>(parsed?.time || "00:00");
  const [errors, setErrors] = useState<{
    scheduleName?: string;
    time?: string;
  }>({});

  const repeatOptions = useMemo(
    () => [
      { value: "daily", label: "Daily" },
      { value: "weekly", label: "Weekly" },
    ],
    [],
  );

  const dayItems = useMemo(
    () => [
      { value: "0", label: "Su" },
      { value: "1", label: "Mo" },
      { value: "2", label: "Tu" },
      { value: "3", label: "We" },
      { value: "4", label: "Th" },
      { value: "5", label: "Fr" },
      { value: "6", label: "Sa" },
    ],
    [],
  );

  function humanizeToCron(): string {
    const [hh, mm] = time.split(":");
    const minute = Number(mm || 0);
    const hour = Number(hh || 0);
    if (repeat === "weekly") {
      const dow = selectedDays.length ? selectedDays.join(",") : "*";
      return `${minute} ${hour} * * ${dow}`;
    }
    return `${minute} ${hour} * * *`;
  }

  const { mutate: updateSchedule, isPending } =
    usePatchV1UpdateExecutionSchedule({
      mutation: {
        onSuccess: async () => {
          invalidateAllScheduleQueries(queryClient, graphId);
          const runsKey = getGetV1ListGraphExecutionsQueryKey(graphId);
          await queryClient.invalidateQueries({ queryKey: runsKey });
          setIsOpen(false);
        },
        onError: (error) => {
          toast({
            title: "Failed to update schedule",
            description:
              error instanceof Error
                ? error.message
                : "An unexpected error occurred.",
            variant: "destructive",
          });
        },
      },
    });

  function handleSave() {
    const errorsNow = validateSchedule({ scheduleName: name, time });
    setErrors(errorsNow);
    if (Object.keys(errorsNow).length > 0) return;
    updateSchedule({
      scheduleId: schedule.id,
      data: { name, cron: humanizeToCron() },
    });
  }

  return {
    isOpen,
    setIsOpen,
    name,
    setName,
    repeat,
    setRepeat,
    selectedDays,
    setSelectedDays,
    time,
    setTime,
    errors,
    repeatOptions,
    dayItems,
    handleSave,
    isPending,
  } as const;
}
