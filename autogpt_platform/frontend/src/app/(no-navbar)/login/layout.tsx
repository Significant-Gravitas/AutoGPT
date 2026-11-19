import { SessionCookieProvider } from "@/components/auth/SessionCookieProvider";
import { hasSessionCookie } from "@/lib/auth/server/hasSessionCookie";
import type { ReactNode } from "react";

export default async function LoginLayout({
  children,
}: {
  children: ReactNode;
}) {
  return (
    <SessionCookieProvider hasSessionCookie={await hasSessionCookie()}>
      {children}
    </SessionCookieProvider>
  );
}
