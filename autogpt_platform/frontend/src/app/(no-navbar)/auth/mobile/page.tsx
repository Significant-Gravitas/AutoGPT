import { AuthCard } from "@/components/auth/AuthCard";
import { Text } from "@/components/atoms/Text/Text";
import {
  mobileAuthConsentPath,
  mobileAuthRequestSchema,
} from "@/lib/auth/mobile-auth-helpers";
import { auth } from "@/lib/auth/auth";
import type { Metadata } from "next";
import { headers } from "next/headers";
import { redirect } from "next/navigation";
import { MobileAuthConsent } from "./components/MobileAuthConsent";

export const metadata: Metadata = {
  title: "Connect AutoGPT mobile",
  robots: { index: false, follow: false },
  referrer: "no-referrer",
};

interface Props {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}

export default async function MobileAuthPage({ searchParams }: Props) {
  const request = mobileAuthRequestSchema.safeParse(await searchParams);
  if (!request.success) {
    return (
      <main className="flex min-h-dvh items-center p-5">
        <AuthCard title="Start sign-in again">
          <Text variant="body" className="text-center">
            This sign-in link is invalid. Open AutoGPT on your phone and start
            sign-in again.
          </Text>
        </AuthCard>
      </main>
    );
  }
  const session = await auth.api.getSession({
    headers: await headers(),
    query: { disableCookieCache: true },
  });
  if (!session) {
    redirect(
      `/login?next=${encodeURIComponent(mobileAuthConsentPath(request.data))}`,
    );
  }
  return (
    <main className="flex min-h-dvh items-center p-5">
      <MobileAuthConsent
        userID={session.user.id}
        email={session.user.email}
        codeChallenge={request.data.code_challenge}
        state={request.data.state}
      />
    </main>
  );
}
