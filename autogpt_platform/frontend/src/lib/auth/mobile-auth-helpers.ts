import { z } from "zod";

export const MOBILE_AUTH_CALLBACK = "autogpt://auth/callback";
export const MOBILE_AUTH_TTL_SECONDS = 90;

export const mobileAuthRequestSchema = z
  .object({
    code_challenge: z.string().regex(/^[A-Za-z0-9_-]{43}$/),
    state: z.string().regex(/^[A-Za-z0-9_-]{32,128}$/),
  })
  .strict();

export const mobileAuthAuthorizeSchema = mobileAuthRequestSchema.extend({
  expected_user_id: z.string().min(1).max(128),
});

export const mobileAuthExchangeSchema = z
  .object({
    code: z.string().regex(/^[A-Za-z0-9_-]{43}$/),
    code_verifier: z.string().regex(/^[A-Za-z0-9._~-]{43,128}$/),
  })
  .strict();

export function mobileAuthConsentPath(
  request: z.infer<typeof mobileAuthRequestSchema>,
) {
  return `/auth/mobile?${new URLSearchParams(request)}`;
}

export function isMobileAuthUserBanned(user: {
  id: string;
  banned?: boolean | null;
  banExpires?: Date | string | null;
}) {
  if (!user.banned) return false;
  if (!user.banExpires) return true;
  const expires = new Date(user.banExpires).getTime();
  return !Number.isFinite(expires) || expires > Date.now();
}

export function isMobileAuthImpersonation(session: {
  userId: string;
  impersonatedBy?: unknown;
}) {
  return Boolean(session.impersonatedBy);
}

export function isMobileAuthSessionBlocked(
  source: {
    user: Parameters<typeof isMobileAuthUserBanned>[0] & {
      emailVerified: boolean;
    };
    session: Parameters<typeof isMobileAuthImpersonation>[0];
  },
  requireEmailVerification: boolean | undefined,
) {
  return (
    isMobileAuthUserBanned(source.user) ||
    isMobileAuthImpersonation(source.session) ||
    Boolean(requireEmailVerification && !source.user.emailVerified)
  );
}
