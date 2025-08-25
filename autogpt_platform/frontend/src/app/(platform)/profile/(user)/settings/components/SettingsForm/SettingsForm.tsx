"use client";

import { Separator } from "@/components/ui/separator";
import { NotificationPreference } from "@/app/api/__generated__/models/notificationPreference";
import { User } from "@supabase/supabase-js";
import { EmailForm } from "./components/EmailForm/EmailForm";
import { NotificationForm } from "./components/NotificationForm/NotificationForm";
import { TimezoneForm } from "./components/TimezoneForm/TimezoneForm";

type SettingsFormProps = {
  preferences: NotificationPreference;
  user: User;
  timezone?: string;
};

export function SettingsForm({
  preferences,
  user,
  timezone,
}: SettingsFormProps) {
  return (
    <div className="flex flex-col gap-8">
      <EmailForm user={user} />
      <Separator />
      <TimezoneForm user={user} currentTimezone={timezone} />
      <Separator />
      <NotificationForm preferences={preferences} user={user} />
    </div>
  );
}
