"use server";

import { revalidatePath } from "next/cache";
import { getServerSupabase } from "@/lib/supabase/server/getServerSupabase";
import { NotificationPreferenceDTO } from "@/lib/autogpt-server-api/types";
import {
  postV1UpdateNotificationPreferences,
  postV1UpdateUserEmail,
} from "@/app/api/__generated__/endpoints/auth/auth";

export async function updateSettings(formData: FormData) {
  const supabase = await getServerSupabase();
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
    await postV1UpdateUserEmail(email);

    if (emailError) {
      throw new Error(`${emailError.message}`);
    }
  }

  try {
    const preferences: NotificationPreferenceDTO = {
      email: user?.email || "",
      preferences: {
        AGENT_RUN: formData.get("notifyOnAgentRun") === "true",
        ZERO_BALANCE: formData.get("notifyOnZeroBalance") === "true",
        LOW_BALANCE: formData.get("notifyOnLowBalance") === "true",
        BLOCK_EXECUTION_FAILED:
          formData.get("notifyOnBlockExecutionFailed") === "true",
        CONTINUOUS_AGENT_ERROR:
          formData.get("notifyOnContinuousAgentError") === "true",
        DAILY_SUMMARY: formData.get("notifyOnDailySummary") === "true",
        WEEKLY_SUMMARY: formData.get("notifyOnWeeklySummary") === "true",
        MONTHLY_SUMMARY: formData.get("notifyOnMonthlySummary") === "true",
      },
      daily_limit: 0,
    };
    await postV1UpdateNotificationPreferences(preferences);
  } catch (error) {
    console.error(error);
    throw new Error(`Failed to update preferences: ${error}`);
  }

  revalidatePath("/profile/settings");
  return { success: true };
}
