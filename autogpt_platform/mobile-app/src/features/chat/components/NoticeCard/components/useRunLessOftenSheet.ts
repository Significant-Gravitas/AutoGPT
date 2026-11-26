import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { ApiError } from "@/lib/api/client";
import { expertActivityKeys } from "@/lib/api/expertActivity";
import { homeKeys } from "@/lib/api/home";
import { getCadence, scheduledChatKeys, setCadence } from "@/lib/api/scheduledChats";

import { cadenceErrorText } from "../../../notice";

interface Args {
  scheduleKey: string;
  onSaved: (label: string) => void;
}

function statusOf(error: unknown) {
  return error instanceof ApiError ? error.status : null;
}

// A refusal the route names answers the same on a second try.
function shouldRetry(failureCount: number, error: unknown) {
  return cadenceErrorText(statusOf(error)) === null && failureCount < 1;
}

export function useRunLessOftenSheet({ scheduleKey, onSaved }: Args) {
  const queryClient = useQueryClient();
  const [selected, setSelected] = useState<string | null>(null);
  const cadence = useQuery({
    queryKey: scheduledChatKeys.cadence(scheduleKey),
    queryFn: () => getCadence(scheduleKey),
    retry: shouldRetry,
  });

  // The sheet is open, so a toast would sit behind it: errors show inside.
  const save = useMutation({
    mutationFn: (cron: string) => setCadence(scheduleKey, cron),
    onSuccess(changed, cron) {
      const chosen = cadence.data?.options.find((option) => option.cron === cron);
      void queryClient.invalidateQueries({ queryKey: expertActivityKeys.graphSchedules });
      void queryClient.invalidateQueries({ queryKey: expertActivityKeys.followupSchedules });
      void queryClient.invalidateQueries({ queryKey: homeKeys.dashboard });
      onSaved(chosen?.label ?? changed.current ?? "less often");
    },
  });

  function choose(cron: string) {
    setSelected(cron);
    save.reset();
  }

  function submit() {
    if (!selected || save.isPending) return;
    save.mutate(selected);
  }

  function retry() {
    void cadence.refetch();
  }

  return {
    cadence: cadence.data ?? null,
    isLoading: cadence.isPending,
    loadError: cadence.isError
      ? (cadenceErrorText(statusOf(cadence.error)) ?? "Couldn't load how often it runs.")
      : null,
    canRetry: cadence.isError && cadenceErrorText(statusOf(cadence.error)) === null,
    retry,
    selected,
    choose,
    submit,
    isSaving: save.isPending,
    saveError: save.isError
      ? (cadenceErrorText(statusOf(save.error)) ?? "Couldn't change how often it runs. Try again.")
      : null,
  };
}
