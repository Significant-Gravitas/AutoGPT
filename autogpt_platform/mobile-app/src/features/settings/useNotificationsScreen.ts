import { useQuery, useQueryClient } from "@tanstack/react-query";
import { router } from "expo-router";
import { useRef } from "react";

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
  // Per cache key, the number of the last save; an older one's answer is stale.
  const lastSave = useRef<Record<string, number>>({});

  // Shown at once, then saved; a refused save puts the old value back. Each
  // save sends every field, so only the newest may write what came back.
  async function save<T>(
    queryKey: readonly string[],
    next: T,
    request: () => Promise<T>,
  ) {
    const id = queryKey.join("/");
    const number = (lastSave.current[id] ?? 0) + 1;
    lastSave.current[id] = number;
    await queryClient.cancelQueries({ queryKey });
    const previous = queryClient.getQueryData<T>(queryKey);
    queryClient.setQueryData<T>(queryKey, next);
    try {
      const answer = await request();
      if (lastSave.current[id] === number) queryClient.setQueryData<T>(queryKey, answer);
    } catch {
      if (lastSave.current[id] !== number) return;
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
    isLoadError:
      (preferences.isError && !preferences.data) || (timezone.isError && !timezone.data),
    isRetrying: preferences.isFetching || timezone.isFetching,
    retry,
    goBack,
    chooseBriefing,
    setAlerts,
    chooseZone,
    toast,
  };
}
