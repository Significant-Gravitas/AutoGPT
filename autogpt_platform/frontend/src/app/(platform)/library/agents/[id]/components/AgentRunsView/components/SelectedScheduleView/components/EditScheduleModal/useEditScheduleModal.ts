"use client";

import { useMemo, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { getGetV1ListExecutionSchedulesForAGraphQueryKey } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { getGetV1ListGraphExecutionsInfiniteQueryOptions } from "@/app/api/__generated__/endpoints/graphs/graphs";
import {
  parseCronToForm,
  validateSchedule,
} from "../../../ScheduleAgentModal/components/ModalScheduleSection/helpers";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { useToast } from "@/components/molecules/Toast/use-toast";

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

  const { mutateAsync, isPending } = useMutation({
    mutationKey: ["patchSchedule", schedule.id],
    mutationFn: async () => {
      const errorsNow = validateSchedule({ scheduleName: name, time });
      setErrors(errorsNow);
      if (Object.keys(errorsNow).length > 0) throw new Error("Invalid form");

      const cron = humanizeToCron();
      const res = await fetch(`/api/schedules/${schedule.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, cron }),
      });
      if (!res.ok) {
        let message = "Failed to update schedule";
        try {
          const data = await res.json();
          message = data?.message || data?.detail || message;
        } catch {
          try {
            message = await res.text();
          } catch {}
        }
        throw new Error(message);
      }
      return res.json();
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: getGetV1ListExecutionSchedulesForAGraphQueryKey(graphId),
      });
      const runsKey = getGetV1ListGraphExecutionsInfiniteQueryOptions(graphId)
        .queryKey as any;
      await queryClient.invalidateQueries({ queryKey: runsKey });
      setIsOpen(false);
    },
    onError: (error: any) => {
      toast({
        title: "❌ Failed to update schedule",
        description: error?.message || "An unexpected error occurred.",
        variant: "destructive",
      });
    },
  });

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
    mutateAsync,
    isPending,
  } as const;
}
