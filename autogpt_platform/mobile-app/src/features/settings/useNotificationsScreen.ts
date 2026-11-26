import { useQuery, useQueryClient } from "@tanstack/react-query";
import { router } from "expo-router";

import { useToast } from "@/components/useToast";
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

import { phoneTimezone, timezoneOffer, withChange } from "./notificationHelpers";

const SAVE_FAILED = "Couldn't save. Try again.";

export function useNotificationsScreen() {
  const queryClient = useQueryClient();
  const { toast, show } = useToast();
  const preferences = useQuery({
    queryKey: preferenceKeys.notifications,
    queryFn: getNotificationPreferences,
  });
  const timezone = useQuery({ queryKey: preferenceKeys.timezone, queryFn: getTimezone });

  // Shown at once, then saved; a refused save puts the old value back.
  async function save<T>(
    queryKey: readonly string[],
    next: T,
    request: () => Promise<T>,
  ) {
    await queryClient.cancelQueries({ queryKey });
    const previous = queryClient.getQueryData<T>(queryKey);
    queryClient.setQueryData<T>(queryKey, next);
    try {
      queryClient.setQueryData<T>(queryKey, await request());
    } catch {
      queryClient.setQueryData<T>(queryKey, previous);
      show({ message: SAVE_FAILED, tone: "error" });
      void queryClient.invalidateQueries({ queryKey });
    }
  }

  function changeNotifications(change: Partial<NotificationPreferences>) {
    const saved = queryClient.getQueryData<NotificationPreferences>(
      preferenceKeys.notifications,
    );
    if (!saved) return;
    const next = withChange(saved, change);
    void save(preferenceKeys.notifications, next, () => updateNotificationPreferences(next));
  }

  function chooseBriefing(frequency: BriefingFrequency) {
    if (frequency === preferences.data?.briefing_frequency) return;
    changeNotifications({ briefing_frequency: frequency });
  }

  function setAlerts(isOn: boolean) {
    changeNotifications({ alerts_enabled: isOn });
  }

  function chooseZone(zone: string) {
    void save<Timezone>(preferenceKeys.timezone, { timezone: zone }, () => updateTimezone(zone));
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
    isLoading: preferences.isPending || timezone.isPending,
    isLoadError:
      (preferences.isError && !preferences.data) || (timezone.isError && !timezone.data),
    retry,
    goBack,
    chooseBriefing,
    setAlerts,
    chooseZone,
    toast,
  };
}
