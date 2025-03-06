"use server";
import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { z } from "zod";
import * as Sentry from "@sentry/nextjs";
import getServerSupabase from "@/lib/supabase/getServerSupabase";
import BackendAPI from "@/lib/autogpt-server-api";
import { loginFormSchema, LoginProvider } from "@/types/auth";

export async function logout() {
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

export async function login(values: z.infer<typeof loginFormSchema>) {
  return await Sentry.withServerActionInstrumentation("login", {}, async () => {
    const supabase = getServerSupabase();
    const api = new BackendAPI();

    if (!supabase) {
      redirect("/error");
    }

    // We are sure that the values are of the correct type because zod validates the form
    const { data, error } = await supabase.auth.signInWithPassword(values);

    if (error) {
      console.error("Error logging in", error);
      return error.message;
    }

    await api.createUser();
    if (!(await api.getUserOnboarding()).isCompleted) {
      revalidatePath("/onboarding", "layout");
      redirect("/onboarding");
    }

    if (data.session) {
      await supabase.auth.setSession(data.session);
    }
    revalidatePath("/", "layout");
    redirect("/");
  });
}

export async function providerLogin(provider: LoginProvider) {
  return await Sentry.withServerActionInstrumentation(
    "providerLogin",
    {},
    async () => {
      const supabase = getServerSupabase();
      const api = new BackendAPI();

      if (!supabase) {
        redirect("/error");
      }

      const { error } = await supabase!.auth.signInWithOAuth({
        provider: provider,
        options: {
          redirectTo:
            process.env.AUTH_CALLBACK_URL ??
            `http://localhost:3000/auth/callback`,
        },
      });

      if (error) {
        console.error("Error logging in", error);
        return error.message;
      }

      await api.createUser();
      if (!(await api.getUserOnboarding()).isCompleted) {
        revalidatePath("/onboarding", "layout");
        redirect("/onboarding");
      }
    },
  );
}
