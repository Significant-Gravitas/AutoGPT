"use client";

import { SettingsForm } from "@/app/(platform)/profile/(user)/settings/components/SettingsForm/SettingsForm";
import { useTimezoneDetection } from "@/app/(platform)/profile/(user)/settings/useTimezoneDetection";
import {
  useGetV1GetNotificationPreferences,
  useGetV1GetUserTimezone,
} from "@/app/api/__generated__/endpoints/auth/auth";
import { okData } from "@/app/api/helpers";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useEffect } from "react";
import SettingsLoading from "./loading";

export default function SettingsPage() {
  const { user } = useSupabase();

  const {
    data: preferences,
    isError: preferencesError,
    isLoading: preferencesLoading,
    error: preferencesErrorData,
    refetch: refetchPreferences,
  } = useGetV1GetNotificationPreferences({
    query: {
      enabled: !!user,
      select: okData,
    },
  });

  const { data: timezone, isLoading: timezoneLoading } =
    useGetV1GetUserTimezone({
      query: {
        enabled: !!user,
        select: (res) => okData(res)?.timezone ?? "not-set",
      },
    });

  useTimezoneDetection(!!user ? timezone : undefined);

  useEffect(() => {
    document.title = "Settings – AutoGPT Platform";
  }, []);

  if (preferencesError) {
    return (
      <div className="container max-w-2xl py-10">
        <ErrorCard
          responseError={
            preferencesErrorData
              ? {
                  detail: preferencesErrorData.detail,
                }
              : undefined
          }
          context="settings"
          onRetry={() => {
            void refetchPreferences();
          }}
        />
      </div>
    );
  }

  if (preferencesLoading || timezoneLoading || !user || !preferences) {
    return <SettingsLoading />;
  }

  return (
    <div className="container max-w-2xl space-y-6 py-10">
      <div className="flex flex-col gap-2">
        <Text variant="h3">My account</Text>
        <Text variant="large">
          Manage your account settings and preferences.
        </Text>
      </div>
      <SettingsForm preferences={preferences} user={user} timezone={timezone} />
    </div>
  );
}
