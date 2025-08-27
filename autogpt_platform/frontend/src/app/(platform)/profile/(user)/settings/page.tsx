"use client";
import {
  useGetV1GetNotificationPreferences,
  useGetV1GetUserTimezone,
} from "@/app/api/__generated__/endpoints/auth/auth";
import { SettingsForm } from "@/app/(platform)/profile/(user)/settings/components/SettingsForm/SettingsForm";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useTimezoneDetection } from "@/hooks/useTimezoneDetection";
import * as React from "react";
import SettingsLoading from "./loading";
import { redirect } from "next/navigation";
import { Text } from "@/components/atoms/Text/Text";

export default function SettingsPage() {
  const {
    data: preferences,
    isError: preferencesError,
    isLoading: preferencesLoading,
  } = useGetV1GetNotificationPreferences({
    query: {
      select: (res) => {
        return res.data;
      },
    },
  });

  const { data: timezoneData, isLoading: timezoneLoading } =
    useGetV1GetUserTimezone({
      query: {
        select: (res) => {
          return res.data;
        },
      },
    });

  const { user, isUserLoading } = useSupabase();

  // Auto-detect timezone if it's not set
  const timezone = timezoneData?.timezone
    ? String(timezoneData.timezone)
    : "not-set";
  useTimezoneDetection(timezone);

  if (preferencesLoading || isUserLoading || timezoneLoading) {
    return <SettingsLoading />;
  }

  if (!user) {
    redirect("/login");
  }

  if (preferencesError || !preferences || !preferences.preferences) {
    return "Errror..."; // TODO: Will use a Error reusable components from Block Menu redesign
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
