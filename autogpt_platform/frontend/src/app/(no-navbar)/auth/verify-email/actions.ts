"use server";
import { auth } from "@/lib/auth/auth";
import {
  expiredLinkLoginPath,
  forgetHeldLink,
  readHeldLink,
  verifyHeldLink,
} from "@/lib/auth/verify-email-confirm";
import * as Sentry from "@sentry/nextjs";
import { cookies, headers } from "next/headers";
import { redirect } from "next/navigation";

export async function confirmEmail() {
  return await Sentry.withServerActionInstrumentation(
    "confirmEmail",
    {},
    async () => {
      const cookieStore = await cookies();
      const link = readHeldLink(cookieStore);
      if (!link) redirect(expiredLinkLoginPath());
      forgetHeldLink(cookieStore);
      redirect(await verifyHeldLink(auth.api, link, await headers()));
    },
  );
}
