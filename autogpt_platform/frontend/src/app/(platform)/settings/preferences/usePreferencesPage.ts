"use client";

import { useEffect, useMemo, useRef, useState } from "react";
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
  const [savedState, setSavedState] = useState<PreferencesFormState>({
    timezone: detectBrowserTimezone(),
    notifications: EMPTY_FLAGS,
  });
  const [isSaving, setIsSaving] = useState(false);
  const hasInitializedFormState = useRef(false);

  useEffect(
    function syncFormStateOnce() {
      if (hasInitializedFormState.current) return;
      if (!preferencesQuery.data || timezoneQuery.data === undefined) return;
      setFormState(initialState);
      setSavedState(initialState);
      hasInitializedFormState.current = true;
    },
    [initialState, preferencesQuery.data, timezoneQuery.data],
  );

  const dirty = useMemo(
    () => isFormDirty(savedState, formState),
    [savedState, formState],
  );

  const dirtyParts = useMemo(
    () => dirtyKinds(savedState, formState),
    [savedState, formState],
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
    setFormState(savedState);
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

      if (dirtyParts.timezone) {
        queryClient.setQueryData(
          getGetV1GetUserTimezoneQueryKey(),
          (prev: unknown) => {
            if (
              prev &&
              typeof prev === "object" &&
              "data" in (prev as Record<string, unknown>)
            ) {
              return {
                ...(prev as Record<string, unknown>),
                data: { timezone: formState.timezone },
              };
            }
            return { status: 200, data: { timezone: formState.timezone } };
          },
        );
      }

      if (dirtyParts.notifications) {
        await queryClient.invalidateQueries({
          queryKey: getGetV1GetNotificationPreferencesQueryKey(),
        });
      }

      setSavedState(formState);

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
    savedState,
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
