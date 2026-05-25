"use client";

import { useEffect, useRef, useState } from "react";
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

  const serverTimezone = timezoneQuery.data ?? "not-set";
  const formTimezone =
    serverTimezone !== "not-set" ? serverTimezone : detectBrowserTimezone();
  const initialFlags = preferencesQuery.data
    ? preferencesToFlags(preferencesQuery.data)
    : EMPTY_FLAGS;
  const initialFormState: PreferencesFormState = {
    timezone: formTimezone,
    notifications: initialFlags,
  };
  const initialSavedState: PreferencesFormState = {
    timezone: serverTimezone,
    notifications: initialFlags,
  };

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
      if (!preferencesQuery.isSuccess) return;
      if (!timezoneQuery.isSuccess) return;
      setFormState(initialFormState);
      setSavedState(initialSavedState);
      hasInitializedFormState.current = true;
    },
    [
      initialFormState,
      initialSavedState,
      preferencesQuery.isSuccess,
      timezoneQuery.isSuccess,
    ],
  );

  const dirty = isFormDirty(savedState, formState);
  const dirtyParts = dirtyKinds(savedState, formState);

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

    const snapshot = formState;
    const partsAtSubmit = dirtyParts;

    setIsSaving(true);

    let timezoneSaved = !partsAtSubmit.timezone;
    let notificationsSaved = !partsAtSubmit.notifications;
    const failures: string[] = [];

    if (partsAtSubmit.timezone) {
      try {
        const result = await updateTimezone.mutateAsync({
          data: {
            timezone: snapshot.timezone as UpdateTimezoneRequestTimezone,
          },
        });
        await queryClient.invalidateQueries({
          queryKey: getGetV1GetUserTimezoneQueryKey(),
        });
        const persistedTimezone =
          (result.status === 200 && result.data?.timezone) || snapshot.timezone;
        setSavedState((prev) => ({ ...prev, timezone: persistedTimezone }));
        timezoneSaved = true;
      } catch (err) {
        failures.push(
          `Time zone: ${err instanceof Error ? err.message : "unknown error"}`,
        );
      }
    }

    if (partsAtSubmit.notifications) {
      try {
        await updateNotifications.mutateAsync({
          data: {
            email: user.email ?? "",
            preferences: flagsToPreferenceMap(snapshot.notifications),
            daily_limit: 0,
          },
        });
        await queryClient.invalidateQueries({
          queryKey: getGetV1GetNotificationPreferencesQueryKey(),
        });
        setSavedState((prev) => ({
          ...prev,
          notifications: snapshot.notifications,
        }));
        notificationsSaved = true;
      } catch (err) {
        failures.push(
          `Notifications: ${err instanceof Error ? err.message : "unknown error"}`,
        );
      }
    }

    setIsSaving(false);

    if (failures.length === 0) {
      toast({ title: "Preferences saved", variant: "success" });
    } else if (timezoneSaved || notificationsSaved) {
      toast({
        title: "Preferences partially saved",
        description: failures.join("; "),
        variant: "destructive",
      });
    } else {
      toast({
        title: "Couldn't save preferences",
        description: failures.join("; "),
        variant: "destructive",
      });
    }
  }

  return {
    user,
    isLoading,
    isError: preferencesQuery.isError || timezoneQuery.isError,
    error: preferencesQuery.error ?? timezoneQuery.error,
    refetch: () => {
      void preferencesQuery.refetch();
      void timezoneQuery.refetch();
    },
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
