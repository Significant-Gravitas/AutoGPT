"use client";
import { useGetV1GetNotificationPreferences } from "@/app/api/__generated__/endpoints/auth/auth";
import { SettingsForm } from "@/app/(platform)/profile/(user)/settings/components/SettingsForm/SettingsForm";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import * as React from "react";
import SettingsLoading from "./loading";
import { redirect } from "next/navigation";

export default function SettingsPage() {
  const {
    data: preferences,
    isError,
    isLoading,
  } = useGetV1GetNotificationPreferences({
    query: {
      select: (res) => {
        return res.data;
      },
    },
  });

  const { user, isUserLoading } = useSupabase();

  if (isLoading || isUserLoading) {
    return <SettingsLoading />;
  }

  if (!user) {
    redirect("/login");
  }

  if (isError || !preferences || !preferences.preferences) {
    return "Errror..."; // TODO: Will use a Error reusable components from Block Menu redesign
  }

  return (
    <div className="container max-w-2xl space-y-6 py-10">
      <div>
        <h3 className="text-lg font-medium">My account</h3>
        <p className="text-sm text-muted-foreground">
          Manage your account settings and preferences.
        </p>
      </div>
      <SettingsForm preferences={preferences} user={user} />
    </div>
  );
}
