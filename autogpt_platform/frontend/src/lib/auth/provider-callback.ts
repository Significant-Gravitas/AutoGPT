import type { LoginProvider } from "@/types/auth";

/** Where a social sign-in returns to. Microsoft is named so /auth/callback
 *  counts the sign-up as Microsoft's rather than Google's. */
export function providerCallbackPath(
  provider: LoginProvider,
  nextUrl: string | null,
): string {
  const params = new URLSearchParams();
  if (provider === "microsoft") params.set("provider", provider);
  if (nextUrl) params.set("next", nextUrl);
  const query = params.toString();
  return query ? `/auth/callback?${query}` : "/auth/callback";
}
