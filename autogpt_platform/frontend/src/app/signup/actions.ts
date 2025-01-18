"use server";
import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { z } from "zod";
import * as Sentry from "@sentry/nextjs";
import getServerSupabase from "@/lib/supabase/getServerSupabase";
import { signupFormSchema } from "@/types/auth";

export async function signup(values: z.infer<typeof signupFormSchema>) {
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
        console.error("Error signing up", error);
        // FIXME: supabase doesn't return the correct error message for this case
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
      redirect("/marketplace/profile");
    },
  );
}
