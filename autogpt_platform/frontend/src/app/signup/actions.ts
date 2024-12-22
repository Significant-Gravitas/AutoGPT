"use server";
import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { z } from "zod";
import * as Sentry from "@sentry/nextjs";
import getServerSupabase from "@/lib/supabase/getServerSupabase";

export const signupFormSchema = z.object({
  email: z.string().email().min(2).max(64),
  password: z.string().min(6).max(64),
  confirmPassword: z.string().min(6).max(64),
  agreeToTerms: z.boolean().refine((value) => value === true, {
    message: "You must agree to the Terms of Use and Privacy Policy",
  }),
}).refine((data) => data.password === data.confirmPassword, {
  message: "Passwords don't match",
  path: ["confirmPassword"],
});

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
