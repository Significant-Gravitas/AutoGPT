"use server";
import getServerSupabase from "@/lib/supabase/getServerSupabase";
import { redirect } from "next/navigation";
import * as Sentry from "@sentry/nextjs";
import { headers } from "next/headers";

export async function sendResetEmail(email: string) {
  return await Sentry.withServerActionInstrumentation(
    "sendResetEmail",
    {},
    async () => {
      const supabase = getServerSupabase();
      const headersList = headers();
      const host = headersList.get("host");
      const protocol =
        process.env.NODE_ENV === "development" ? "http" : "https";
      const origin = `${protocol}://${host}`;

      if (!supabase) {
        redirect("/error");
      }

      const { error } = await supabase.auth.resetPasswordForEmail(email, {
        redirectTo: `${origin}/reset_password`,
      });

      if (error) {
        console.error("Error sending reset email", error);
        return error.message;
      }

      redirect("/reset_password");
    },
  );
}

export async function changePassword(password: string) {
  return await Sentry.withServerActionInstrumentation(
    "changePassword",
    {},
    async () => {
      const supabase = getServerSupabase();

      if (!supabase) {
        redirect("/error");
      }

      const { error } = await supabase.auth.updateUser({ password });

      if (error) {
        console.error("Error changing password", error);
        return error.message;
      }

      await supabase.auth.signOut();
      redirect("/login");
    },
  );
}
