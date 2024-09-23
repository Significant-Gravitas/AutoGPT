"use server";
import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { createServerClient } from "@/lib/supabase/server";
import { z } from "zod";
import * as Sentry from "@sentry/nextjs";

const loginFormSchema = z.object({
  email: z.string().email().min(2).max(64),
  password: z.string().min(6).max(64),
});

export async function login(values: z.infer<typeof loginFormSchema>) {
  return await Sentry.withServerActionInstrumentation("login", {}, async () => {
    const supabase = createServerClient();

    if (!supabase) {
      redirect("/error");
    }

    // We are sure that the values are of the correct type because zod validates the form
    const { data, error } = await supabase.auth.signInWithPassword(values);

    if (error) {
      return error.message;
    }

    if (data.session) {
      await supabase.auth.setSession(data.session);
    }

    revalidatePath("/", "layout");
    redirect("/profile");
  });
}

export async function signup(values: z.infer<typeof loginFormSchema>) {
  "use server";
  return await Sentry.withServerActionInstrumentation(
    "signup",
    {},
    async () => {
      const supabase = createServerClient();

      if (!supabase) {
        redirect("/error");
      }

      // We are sure that the values are of the correct type because zod validates the form
      const { data, error } = await supabase.auth.signUp(values);

      if (error) {
        return error.message;
      }

      if (data.session) {
        await supabase.auth.setSession(data.session);
      }

      revalidatePath("/", "layout");
      redirect("/profile");
    },
  );
}
