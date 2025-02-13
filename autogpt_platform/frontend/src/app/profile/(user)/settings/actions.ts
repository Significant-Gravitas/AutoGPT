"use server";

import { revalidatePath } from "next/cache";
import getServerSupabase from "@/lib/supabase/getServerSupabase";

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

    if (emailError) {
      throw new Error(`${emailError.message}`);
    }
  }

  // TODO: @ntindle Handle updating notification preferences here
  // const preferencesError = {};
  //
  // if (preferencesError) {
  //   throw new SettingsError(
  //     `Failed to update preferences: ${preferencesError.message}`,
  //   );
  // }

  revalidatePath("/profile/settings");
  return { success: true };
}
