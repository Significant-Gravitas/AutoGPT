"use client";

import { getGetV1ListGraphExecutionsQueryKey } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { usePatchV1UpdateExecutionSchedule } from "@/app/api/__generated__/endpoints/schedules/schedules";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { safeHumanizeCronExpression } from "@/lib/cron-expression-utils";
import { invalidateAllScheduleQueries } from "@/services/schedules/invalidate-schedules";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { validateSchedule } from "../../../../modals/ScheduleAgentModal/components/ModalScheduleSection/helpers";
import {
  buildCron,
  DAY_ITEMS,
  type EditScheduleForm,
  formFromSchedule,
  REPEAT_OPTIONS,
} from "./helpers";

interface FormErrors {
  scheduleName?: string;
  time?: string;
  days?: string;
}

export function useEditScheduleModal(
  graphId: string,
  schedule: GraphExecutionJobInfo,
) {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [isOpen, setIsOpen] = useState(false);
  const [form, setForm] = useState(() => formFromSchedule(schedule));
  const [errors, setErrors] = useState<FormErrors>({});

  function update(fields: Partial<EditScheduleForm>) {
    setForm((current) => ({ ...current, ...fields }));
  }

  // The dialog stays mounted while the selected schedule changes, so the form
  // reloads from whichever schedule is current each time it opens.
  function handleOpenChange(open: boolean) {
    if (open) {
      setForm(formFromSchedule(schedule));
      setErrors({});
    }
    setIsOpen(open);
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
    const errorsNow: FormErrors = validateSchedule({
      scheduleName: form.name,
      time: form.canEditTiming ? form.time : undefined,
    });
    if (
      form.canEditTiming &&
      form.repeat === "weekly" &&
      form.selectedDays.length === 0
    ) {
      errorsNow.days = "Pick at least one day";
    }
    setErrors(errorsNow);
    if (Object.keys(errorsNow).length > 0) return;
    updateSchedule({
      scheduleId: schedule.id,
      data: form.canEditTiming
        ? { name: form.name, cron: buildCron(form) }
        : { name: form.name },
    });
  }

  return {
    isOpen,
    setIsOpen: handleOpenChange,
    name: form.name,
    setName: (name: string) => update({ name }),
    canEditTiming: form.canEditTiming,
    recurrenceLabel: safeHumanizeCronExpression(schedule.cron),
    repeat: form.repeat,
    setRepeat: (repeat: string) => update({ repeat }),
    selectedDays: form.selectedDays,
    setSelectedDays: (selectedDays: string[]) => update({ selectedDays }),
    time: form.time,
    setTime: (time: string) => update({ time }),
    errors,
    repeatOptions: REPEAT_OPTIONS,
    dayItems: DAY_ITEMS,
    handleSave,
    isPending,
  };
}
