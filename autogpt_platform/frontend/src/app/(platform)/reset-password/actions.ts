"use server";
import * as Sentry from "@sentry/nextjs";
import { redirect } from "next/navigation";
import { cookies } from "next/headers";
import { environment } from "@/services/environment";

export async function sendResetEmail(email: string) {
  return await Sentry.withServerActionInstrumentation(
    "sendResetEmail",
    {},
    async () => {
      const origin =
        process.env.NEXT_PUBLIC_FRONTEND_BASE_URL || "http://localhost:3000";

      try {
        const response = await fetch(
          `${environment.getAGPTServerBaseUrl()}/api/auth/password-reset/request`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              email,
              redirect_url: `${origin}/api/auth/callback/reset-password`,
            }),
          },
        );

        if (!response.ok) {
          const data = await response.json();
          console.error("Error sending reset email", data);
          return data.detail || "Failed to send reset email";
        }
      } catch (error) {
        console.error("Error sending reset email", error);
        return "Failed to send reset email";
      }
    },
  );
}

export async function changePassword(password: string) {
  return await Sentry.withServerActionInstrumentation(
    "changePassword",
    {},
    async () => {
      const cookieStore = await cookies();
      const resetToken = cookieStore.get("password_reset_token")?.value;

      if (!resetToken) {
        return "Invalid or expired reset link. Please request a new password reset.";
      }

      try {
        const response = await fetch(
          `${environment.getAGPTServerBaseUrl()}/api/auth/password-reset/confirm`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              token: resetToken,
              new_password: password,
            }),
          },
        );

        if (!response.ok) {
          const data = await response.json();
          console.error("Error changing password", data);
          return data.detail || "Failed to change password";
        }

        // Clear the reset token cookie
        cookieStore.delete("password_reset_token");
      } catch (error) {
        console.error("Error changing password", error);
        return "Failed to change password";
      }

      redirect("/login");
    },
  );
}
