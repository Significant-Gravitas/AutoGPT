import { getGetHomeDashboardQueryKey } from "@/app/api/__generated__/endpoints/home/home";
import {
  getGetScheduledChatCadenceQueryKey,
  useGetScheduledChatCadence,
  useSetScheduledChatCadence,
} from "@/app/api/__generated__/endpoints/schedules/schedules";
import { okData } from "@/app/api/helpers";
import { toast } from "@/components/molecules/Toast/use-toast";
import { invalidateAllScheduleQueries } from "@/services/schedules/invalidate-schedules";
import * as Sentry from "@sentry/nextjs";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { changeErrorText, lowerFirst } from "./helpers";

interface Args {
  scheduleKey: string;
  isOpen: boolean;
  onDone: () => void;
}

export function useRunLessOftenDialog({ scheduleKey, isOpen, onDone }: Args) {
  const queryClient = useQueryClient();
  const [selected, setSelected] = useState<string | null>(null);

  const cadenceQuery = useGetScheduledChatCadence(scheduleKey, {
    query: { enabled: isOpen, select: (res) => okData(res) },
  });
  const cadence = cadenceQuery.data;

  const { mutate, isPending } = useSetScheduledChatCadence({
    mutation: {
      onSuccess: (res) => {
        const changed = okData(res);
        queryClient.invalidateQueries({
          queryKey: getGetScheduledChatCadenceQueryKey(scheduleKey),
        });
        queryClient.invalidateQueries({
          queryKey: getGetHomeDashboardQueryKey(),
        });
        void invalidateAllScheduleQueries(queryClient);
        toast({
          title: "It runs less often now",
          description:
            changed?.current != null
              ? `“${changed.title}” now runs ${lowerFirst(changed.current)}.`
              : undefined,
        });
        setSelected(null);
        onDone();
      },
      onError: (error) => {
        const known = changeErrorText(error);
        if (known === null) Sentry.captureException(error);
        toast({
          variant: "destructive",
          title: "Couldn't change how often it runs",
          description: known ?? "Please try again.",
        });
      },
    },
  });

  function save() {
    if (!selected || isPending) return;
    mutate({ scheduleKey, data: { cron: selected } });
  }

  return {
    cadence,
    isLoading: cadenceQuery.isLoading,
    isError: cadenceQuery.isError,
    selected,
    select: setSelected,
    save,
    isSaving: isPending,
  };
}
