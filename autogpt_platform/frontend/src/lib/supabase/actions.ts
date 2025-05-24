"use server";
import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import * as Sentry from "@sentry/nextjs";

import getServerSupabase from "@/lib/supabase/getServerSupabase";

export async function _logoutServer() {
  return await Sentry.withServerActionInstrumentation(
    "logout",
    {},
    async () => {
      const supabase = getServerSupabase();

      if (!supabase) {
        redirect("/error");
      }

      const { error } = await supabase.auth.signOut();

      if (error) {
        console.error("Error logging out", error);
        return error.message;
      }

      revalidatePath("/", "layout");
      redirect("/login");
    },
  );
}
