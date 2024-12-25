"use server";
import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { z } from "zod";
import * as Sentry from "@sentry/nextjs";
import getServerSupabase from "@/lib/supabase/getServerSupabase";
import BackendAPI from "@/lib/autogpt-server-api";

const loginFormSchema = z.object({
  email: z.string().email().min(2).max(64),
  password: z.string().min(6).max(64),
});

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
        console.log("Error logging out", error);
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

    await api.createUser();

    if (error) {
      console.log("Error logging in", error);
      if (error.status == 400) {
        // Hence User is not present
        redirect("/login");
      }

      return error.message;
    }

    if (data.session) {
      await supabase.auth.setSession(data.session);
    }
    console.log("Logged in");
    revalidatePath("/", "layout");
    redirect("/");
  });
}

export async function signup(values: z.infer<typeof loginFormSchema>) {
  "use server";
  return await Sentry.withServerActionInstrumentation(
    "signup",
    {},
    async () => {
      const supabase = getServerSupabase();

      if (!supabase) {
        redirect("/error");
      }

      // We are sure that the values are of the correct type because zod validates the form
      const { data, error } = await supabase.auth.signUp(values);

      if (error) {
        console.log("Error signing up", error);
        if (error.message.includes("P0001")) {
          return "Please join our waitlist for your turn: https://agpt.co/waitlist";
        }
        if (error.code?.includes("user_already_exists")) {
          redirect("/login");
        }
        return error.message;
      }

      if (data.session) {
        await supabase.auth.setSession(data.session);
      }
      console.log("Signed up");
      revalidatePath("/", "layout");
      redirect("/store/profile");
    },
  );
}
