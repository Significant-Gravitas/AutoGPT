"use client";

import { useEffect, useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import {
  getGetV1GetNotificationPreferencesQueryKey,
  getGetV1GetUserTimezoneQueryKey,
  useGetV1GetNotificationPreferences,
  useGetV1GetUserTimezone,
  usePostV1UpdateNotificationPreferences,
  usePostV1UpdateUserTimezone,
} from "@/app/api/__generated__/endpoints/auth/auth";
import type { UpdateTimezoneRequestTimezone } from "@/app/api/__generated__/models/updateTimezoneRequestTimezone";
import { okData } from "@/app/api/helpers";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";

import {
  detectBrowserTimezone,
  dirtyKinds,
  flagsToPreferenceMap,
  isFormDirty,
  preferencesToFlags,
  type NotificationFlags,
  type NotificationKey,
  type PreferencesFormState,
} from "./helpers";

const EMPTY_FLAGS: NotificationFlags = {
  notifyOnAgentRun: false,
  notifyOnBlockExecutionFailed: false,
  notifyOnContinuousAgentError: false,
  notifyOnAgentApproved: false,
  notifyOnAgentRejected: false,
  notifyOnZeroBalance: false,
  notifyOnLowBalance: false,
  notifyOnDailySummary: false,
  notifyOnWeeklySummary: false,
  notifyOnMonthlySummary: false,
};

export function usePreferencesPage() {
  const { user } = useSupabase();
  const queryClient = useQueryClient();

  const preferencesQuery = useGetV1GetNotificationPreferences({
    query: {
      enabled: !!user,
      select: okData,
    },
  });

  const timezoneQuery = useGetV1GetUserTimezone({
    query: {
      enabled: !!user,
      select: (res) => okData(res)?.timezone ?? "not-set",
    },
  });

  const isLoading =
    !user ||
    preferencesQuery.isLoading ||
    timezoneQuery.isLoading ||
    !preferencesQuery.data;

  const initialState = useMemo<PreferencesFormState>(() => {
    const tz =
      timezoneQuery.data && timezoneQuery.data !== "not-set"
        ? timezoneQuery.data
        : detectBrowserTimezone();
    const flags = preferencesQuery.data
      ? preferencesToFlags(preferencesQuery.data)
      : EMPTY_FLAGS;
    return { timezone: tz, notifications: flags };
  }, [preferencesQuery.data, timezoneQuery.data]);

  const [formState, setFormState] = useState<PreferencesFormState>({
    timezone: detectBrowserTimezone(),
    notifications: EMPTY_FLAGS,
  });
  const [isSaving, setIsSaving] = useState(false);

  useEffect(
    function syncFormStateOnLoad() {
      if (!preferencesQuery.data) return;
      setFormState(initialState);
    },
    [initialState, preferencesQuery.data],
  );

  const dirty = useMemo(
    () => isFormDirty(initialState, formState),
    [initialState, formState],
  );

  const dirtyParts = useMemo(
    () => dirtyKinds(initialState, formState),
    [initialState, formState],
  );

  function setTimezone(value: string) {
    setFormState((prev) => ({ ...prev, timezone: value }));
  }

  function toggleNotification(key: NotificationKey, value: boolean) {
    setFormState((prev) => ({
      ...prev,
      notifications: { ...prev.notifications, [key]: value },
    }));
  }

  function setAllInGroup(keys: NotificationKey[], value: boolean) {
    setFormState((prev) => {
      const next = { ...prev.notifications };
      for (const key of keys) next[key] = value;
      return { ...prev, notifications: next };
    });
  }

  function discardChanges() {
    setFormState(initialState);
  }

  const updateTimezone = usePostV1UpdateUserTimezone();
  const updateNotifications = usePostV1UpdateNotificationPreferences();

  async function savePreferences() {
    if (!dirty || isSaving || !user) return;

    setIsSaving(true);
    try {
      const tasks: Promise<unknown>[] = [];

      if (dirtyParts.timezone) {
        tasks.push(
          updateTimezone.mutateAsync({
            data: {
              timezone:
                formState.timezone as UpdateTimezoneRequestTimezone,
            },
          }),
        );
      }

      if (dirtyParts.notifications) {
        tasks.push(
          updateNotifications.mutateAsync({
            data: {
              email: user.email ?? "",
              preferences: flagsToPreferenceMap(formState.notifications),
              daily_limit: 0,
            },
          }),
        );
      }

      await Promise.all(tasks);

      await Promise.all([
        dirtyParts.timezone
          ? queryClient.invalidateQueries({
              queryKey: getGetV1GetUserTimezoneQueryKey(),
            })
          : null,
        dirtyParts.notifications
          ? queryClient.invalidateQueries({
              queryKey: getGetV1GetNotificationPreferencesQueryKey(),
            })
          : null,
      ]);

      toast({ title: "Preferences saved", variant: "success" });
    } catch (err) {
      toast({
        title: "Couldn't save preferences",
        description: err instanceof Error ? err.message : undefined,
        variant: "destructive",
      });
    } finally {
      setIsSaving(false);
    }
  }

  return {
    user,
    isLoading,
    isError: preferencesQuery.isError,
    error: preferencesQuery.error,
    refetch: preferencesQuery.refetch,
    formState,
    initialState,
    rawTimezone: timezoneQuery.data,
    dirty,
    isSaving,
    setTimezone,
    toggleNotification,
    setAllInGroup,
    discardChanges,
    savePreferences,
  };
}
