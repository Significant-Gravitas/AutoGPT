"use server";
import { getServerSupabase } from "@/lib/supabase/server/getServerSupabase";
import * as Sentry from "@sentry/nextjs";
import { redirect } from "next/navigation";

export async function sendResetEmail(email: string) {
  return await Sentry.withServerActionInstrumentation(
    "sendResetEmail",
    {},
    async () => {
      const supabase = await getServerSupabase();
      const origin =
        process.env.NEXT_PUBLIC_FRONTEND_BASE_URL || "http://localhost:3000";

      if (!supabase) {
        redirect("/error");
      }

      const { error } = await supabase.auth.resetPasswordForEmail(email, {
        redirectTo: `${origin}/api/auth/callback/reset-password`,
      });

      if (error) {
        console.error("Error sending reset email", error);
        return error.message;
      }
    },
  );
}

export async function changePassword(password: string) {
  return await Sentry.withServerActionInstrumentation(
    "changePassword",
    {},
    async () => {
      const supabase = await getServerSupabase();

      if (!supabase) {
        redirect("/error");
      }

      const { error } = await supabase.auth.updateUser({ password });

      if (error) {
        console.error("Error changing password", error);
        return error.message;
      }

      await supabase.auth.signOut({ scope: "global" });
      redirect("/login");
    },
  );
}
