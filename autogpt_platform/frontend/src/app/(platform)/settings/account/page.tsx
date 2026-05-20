"use client";

import { useEffect } from "react";

import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";

import { AccountCard } from "./components/AccountCard/AccountCard";
import { PreferencesHeader } from "./components/PreferencesHeader/PreferencesHeader";
import { PreferencesSkeleton } from "./components/PreferencesSkeleton/PreferencesSkeleton";
import { SaveBar } from "./components/SaveBar/SaveBar";
import { TimezoneCard } from "./components/TimezoneCard/TimezoneCard";
import { usePreferencesPage } from "./usePreferencesPage";

export default function SettingsPreferencesPage() {
  useEffect(() => {
    document.title = "Account – AutoGPT Platform";
  }, []);

  const {
    user,
    isLoading,
    isError,
    error,
    refetch,
    formState,
    dirty,
    isSaving,
    setTimezone,
    discardChanges,
    savePreferences,
  } = usePreferencesPage();

  if (isError) {
    return (
      <ErrorCard
        context="settings"
        responseError={
          error ? { detail: (error as { detail?: string }).detail } : undefined
        }
        onRetry={() => {
          void refetch();
        }}
      />
    );
  }

  if (isLoading || !user) {
    return <PreferencesSkeleton />;
  }

  return (
    <div className="flex flex-col gap-6 pb-8">
      <PreferencesHeader />

      <AccountCard user={user} index={0} />

      <TimezoneCard
        value={formState.timezone}
        onChange={setTimezone}
        index={1}
      />

      <SaveBar
        visible={dirty}
        saving={isSaving}
        onDiscard={discardChanges}
        onSave={savePreferences}
      />
    </div>
  );
}
