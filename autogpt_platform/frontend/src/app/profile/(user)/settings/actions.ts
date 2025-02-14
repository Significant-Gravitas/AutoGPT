"use server";

import { revalidatePath } from "next/cache";
import getServerSupabase from "@/lib/supabase/getServerSupabase";
import BackendApi from "@/lib/autogpt-server-api";
import { NotificationPreferenceDTO } from "@/lib/autogpt-server-api/types";

export async function updateSettings(formData: FormData) {
  const supabase = getServerSupabase();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  // Handle auth-related updates
  const password = formData.get("password") as string;
  const email = formData.get("email") as string;

  if (password) {
    const { error: passwordError } = await supabase.auth.updateUser({
      password,
    });

    if (passwordError) {
      throw new Error(`${passwordError.message}`);
    }
  }

  if (email !== user?.email) {
    const { error: emailError } = await supabase.auth.updateUser({
      email,
    });
    const api = new BackendApi();
    await api.updateUserEmail(email);

    if (emailError) {
      throw new Error(`${emailError.message}`);
    }
  }

  // const preferencesError = {};
  // if (preferencesError) {
  //   throw new SettingsError(
  //     `Failed to update preferences: ${preferencesError.message}`,
  //   );
  // }

  const api = new BackendApi();
  const preferences: NotificationPreferenceDTO = {
    email: user?.email || "",
    preferences: {
      agent_run: formData.get("notifyOnAgentRun") === "true",
      zero_balance: formData.get("notifyOnZeroBalance") === "true",
      low_balance: formData.get("notifyOnLowBalance") === "true",
      block_execution_failed:
        formData.get("notifyOnBlockExecutionFailed") === "true",
      continuous_agent_error:
        formData.get("notifyOnContinuousAgentError") === "true",
      daily_summary: formData.get("notifyOnDailySummary") === "true",
      weekly_summary: formData.get("notifyOnWeeklySummary") === "true",
      monthly_summary: formData.get("notifyOnMonthlySummary") === "true",
    },
    daily_limit: 0,
  };
  await api.updateUserPreferences(preferences);

  revalidatePath("/profile/settings");
  return { success: true };
}
