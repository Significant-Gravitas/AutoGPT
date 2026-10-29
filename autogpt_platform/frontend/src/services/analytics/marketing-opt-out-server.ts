import * as Sentry from "@sentry/nextjs";
import { cookies } from "next/headers";
import { MARKETING_OPT_OUT_COOKIE } from "./marketing-opt-out-cookie";

// Reads and clears the signup-page marketing refusal carried across the Google
// OAuth round trip (see marketing-opt-out-cookie.ts). A missing cookie means
// the person did not opt out. Best effort: an unreadable cookie counts as not
// refused rather than failing the sign-in.
export async function takeMarketingOptOutFlag(): Promise<boolean> {
  try {
    const cookieStore = await cookies();
    const flag = cookieStore.get(MARKETING_OPT_OUT_COOKIE);
    if (!flag) return false;

    cookieStore.delete(MARKETING_OPT_OUT_COOKIE);
    return flag.value === "1";
  } catch (error) {
    Sentry.captureException(error, {
      tags: { signup_step: "read_marketing_opt_out" },
    });
    return false;
  }
}
