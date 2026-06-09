"use server";
import { auth } from "@/lib/auth/auth";
import * as Sentry from "@sentry/nextjs";
import { redirect } from "next/navigation";
import { headers } from "next/headers";

export async function sendResetEmail(email: string) {
  return await Sentry.withServerActionInstrumentation(
    "sendResetEmail",
    {},
    async () => {
      const supabase = await getServerSupabase();
      const origin =
        process.env.NEXT_PUBLIC_FRONTEND_BASE_URL || "http://localhost:3000";
      try {
        await auth.api.requestPasswordReset({
          body: {
            email,
            redirectTo: `${origin}/api/auth/callback/reset-password`,
          },
          headers: new Headers(await headers()),
        });
      } catch (error) {
        console.error("Error sending reset email", error);
        return error instanceof Error ? error.message : "Failed to send reset email";
      }
    },
  );
}

export async function changePassword(password: string, token?: string | null) {
  return await Sentry.withServerActionInstrumentation(
    "changePassword",
    {},
    async () => {
      if (!token) {
        return "Missing password reset token";
      }

      try {
        await auth.api.resetPassword({
          body: {
            newPassword: password,
            token,
          },
          headers: new Headers(await headers()),
        });
      } catch (error) {
        console.error("Error changing password", error);
        return error instanceof Error ? error.message : "Failed to change password";
      }

      redirect("/login");
    },
  );
}
