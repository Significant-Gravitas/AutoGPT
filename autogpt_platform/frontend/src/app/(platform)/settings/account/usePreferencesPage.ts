"use client";

import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
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
import { useAuth } from "@/lib/auth/hooks/useAuth";

import {
  detectBrowserTimezone,
  dirtyKinds,
  isFormDirty,
  preferencesToSettings,
  settingsFromFooterLink,
  type BriefingFrequency,
  type NotificationSettings,
  type PreferencesFormState,
} from "./helpers";

const DEFAULT_SETTINGS: NotificationSettings = {
  briefingFrequency: "WEEKLY",
  alertsEnabled: true,
  storeVerdictsEnabled: true,
};

export function usePreferencesPage() {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  // The Briefing footer's volume knob links here with ?f=daily|weekly|monthly
  // |alerts|off, and the choice is applied on load so it really is one click.
  const footerChoice = useSearchParams().get("f");

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
  const initialSettings = preferencesQuery.data
    ? preferencesToSettings(preferencesQuery.data)
    : DEFAULT_SETTINGS;
  const initialFormState: PreferencesFormState = {
    timezone: formTimezone,
    notifications:
      settingsFromFooterLink(footerChoice, initialSettings) ?? initialSettings,
  };
  const initialSavedState: PreferencesFormState = {
    timezone: serverTimezone,
    notifications: initialSettings,
  };

  const [formState, setFormState] = useState<PreferencesFormState>({
    timezone: detectBrowserTimezone(),
    notifications: DEFAULT_SETTINGS,
  });
  const [savedState, setSavedState] = useState<PreferencesFormState>({
    timezone: detectBrowserTimezone(),
    notifications: DEFAULT_SETTINGS,
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

  function setBriefingFrequency(value: BriefingFrequency) {
    setFormState((prev) => ({
      ...prev,
      notifications: { ...prev.notifications, briefingFrequency: value },
    }));
  }

  function setAlertsEnabled(value: boolean) {
    setFormState((prev) => ({
      ...prev,
      notifications: { ...prev.notifications, alertsEnabled: value },
    }));
  }

  function setStoreVerdictsEnabled(value: boolean) {
    setFormState((prev) => ({
      ...prev,
      notifications: { ...prev.notifications, storeVerdictsEnabled: value },
    }));
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
            briefing_frequency: snapshot.notifications.briefingFrequency,
            alerts_enabled: snapshot.notifications.alertsEnabled,
            store_verdicts_enabled:
              snapshot.notifications.storeVerdictsEnabled,
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
    setBriefingFrequency,
    setAlertsEnabled,
    setStoreVerdictsEnabled,
    discardChanges,
    savePreferences,
  };
}
