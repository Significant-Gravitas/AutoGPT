import * as React from "react";
import { Metadata } from "next";
import SettingsForm from "@/components/profile/settings/SettingsForm";
import getServerUser from "@/lib/supabase/getServerUser";
import { redirect } from "next/navigation";
import { getUserPreferences } from "./actions";
export const metadata: Metadata = {
  title: "Settings",
  description: "Manage your account settings and preferences.",
};

export default async function SettingsPage() {
  const { user, error } = await getServerUser();

  if (error || !user) {
    redirect("/login");
  }

  const preferences = await getUserPreferences();

  return (
    <div className="space-y-6 pb-10">
      {/* Title */}
      <h1 className="font-poppins text-[1.75rem] font-medium leading-[2.5rem] text-zinc-500">
        Settings
      </h1>

      <SettingsForm user={user} preferences={preferences} />
    </div>
  );
}
