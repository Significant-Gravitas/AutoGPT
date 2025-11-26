"use client";
import { SettingsForm } from "@/app/(platform)/profile/(user)/settings/components/SettingsForm/SettingsForm";
import { useTimezoneDetection } from "@/app/(platform)/profile/(user)/settings/useTimezoneDetection";
import {
  useGetV1GetNotificationPreferences,
  useGetV1GetUserTimezone,
} from "@/app/api/__generated__/endpoints/auth/auth";
import { Text } from "@/components/atoms/Text/Text";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { redirect } from "next/navigation";
import { useEffect } from "react";
import SettingsLoading from "./loading";

export default function SettingsPage() {
  const {
    data: preferences,
    isError: preferencesError,
    isLoading: preferencesLoading,
  } = useGetV1GetNotificationPreferences({
    query: { select: (res) => (res.status === 200 ? res.data : null) },
  });

  const { data: timezone, isLoading: timezoneLoading } =
    useGetV1GetUserTimezone({
      query: {
        select: (res) => {
          return res.status === 200 ? String(res.data.timezone) : "not-set";
        },
      },
    });

  useTimezoneDetection(timezone);

  const { user, isUserLoading } = useSupabase();

  useEffect(() => {
    document.title = "Settings – AutoGPT Platform";
  }, []);

  if (preferencesLoading || isUserLoading || timezoneLoading) {
    return <SettingsLoading />;
  }

  if (!user) {
    redirect("/login");
  }

  if (preferencesError || !preferences || !preferences.preferences) {
    return "Error..."; // TODO: Will use a Error reusable components from Block Menu redesign
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
