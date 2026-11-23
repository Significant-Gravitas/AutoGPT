import {
  expiredLinkLoginPath,
  heldLinkAddress,
  readHeldLink,
} from "@/lib/auth/verify-email-confirm";
import type { Metadata } from "next";
import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { ConfirmEmail } from "./components/ConfirmEmail";

export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Confirm your email",
  robots: { index: false, follow: false },
};

export default async function VerifyEmailPage() {
  const link = readHeldLink(await cookies());
  if (!link) redirect(expiredLinkLoginPath());

  return <ConfirmEmail email={heldLinkAddress(link)} />;
}
