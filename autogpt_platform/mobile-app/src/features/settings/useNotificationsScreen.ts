import { useMutation, useMutationState, useQuery, useQueryClient } from "@tanstack/react-query";
import { router } from "expo-router";

import {
  type BriefingFrequency,
  getNotificationPreferences,
  getTimezone,
  type NotificationPreferences,
  preferenceKeys,
  type Timezone,
  updateNotificationPreferences,
  updateTimezone,
} from "@/lib/api/preferences";
import { sessionGeneration } from "@/lib/session";

import { phoneTimezone, timezoneOffer, withChange } from "./notificationHelpers";

function usePreferenceSave<T>(
  queryKey: readonly string[],
  request: (next: T) => Promise<T>,
) {
  const queryClient = useQueryClient();
  const generation = sessionGeneration();
  const mutationKey = ["savePreference", ...queryKey, generation];
  const status = useMutationState({
    filters: { mutationKey, exact: true },
    select: (entry) => entry.state.status,
  }).at(-1);
  const mutation = useMutation({
    mutationKey,
    async mutationFn(next: T) {
      await queryClient.cancelQueries({ queryKey });
      if (generation !== sessionGeneration()) return;
      const previous = queryClient.getQueryData<T>(queryKey);
      queryClient.setQueryData<T>(queryKey, next);
      try {
        const answer = await request(next);
        if (generation !== sessionGeneration()) return;
        await queryClient.cancelQueries({ queryKey });
        if (generation === sessionGeneration()) queryClient.setQueryData<T>(queryKey, answer);
      } catch (error) {
        if (generation !== sessionGeneration()) return;
        queryClient.setQueryData<T>(queryKey, previous);
        void queryClient.invalidateQueries({ queryKey });
        throw error;
      }
    },
  });

  function save(next: T) {
    if (generation !== sessionGeneration() || queryClient.isMutating({ mutationKey })) return;
    mutation.mutate(next);
  }

  return { isSaving: status === "pending", isSaveError: status === "error", save };
}

function saveTimezone(next: Timezone) {
  return updateTimezone(next.timezone);
}

export function useNotificationsScreen() {
  const queryClient = useQueryClient();
  const preferences = useQuery({
    queryKey: preferenceKeys.notifications,
    queryFn: getNotificationPreferences,
  });
  const timezone = useQuery({ queryKey: preferenceKeys.timezone, queryFn: getTimezone });
  const preferencesSave = usePreferenceSave(
    preferenceKeys.notifications,
    updateNotificationPreferences,
  );
  const timezoneSave = usePreferenceSave(preferenceKeys.timezone, saveTimezone);

  function changeNotifications(change: Partial<NotificationPreferences>) {
    const saved = queryClient.getQueryData<NotificationPreferences>(
      preferenceKeys.notifications,
    );
    if (!saved) return;
    preferencesSave.save(withChange(saved, change));
  }

  function chooseBriefing(frequency: BriefingFrequency) {
    if (frequency === preferences.data?.briefing_frequency) return;
    changeNotifications({ briefing_frequency: frequency });
  }

  function setAlerts(isOn: boolean) {
    changeNotifications({ alerts_enabled: isOn });
  }

  function chooseZone(zone: string) {
    timezoneSave.save({ timezone: zone });
  }

  function retry() {
    void preferences.refetch();
    void timezone.refetch();
  }

  function goBack() {
    router.navigate("/settings");
  }

  const zone = timezone.data?.timezone ?? null;

  return {
    preferences: preferences.data ?? null,
    zone,
    zoneOffer: zone === null ? null : timezoneOffer(zone, phoneTimezone()),
    isLoadError:
      (preferences.isError && !preferences.data) || (timezone.isError && !timezone.data),
    isRetrying: preferences.isFetching || timezone.isFetching,
    isSavingPreferences: preferencesSave.isSaving,
    isSavingTimezone: timezoneSave.isSaving,
    isSaveError: preferencesSave.isSaveError || timezoneSave.isSaveError,
    retry,
    goBack,
    chooseBriefing,
    setAlerts,
    chooseZone,
  };
}
