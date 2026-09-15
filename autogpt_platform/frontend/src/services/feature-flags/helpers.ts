import type { User } from "@/lib/auth/types";

export type LDUserContext =
  | {
      kind: "user";
      key: string;
      anonymous: true;
    }
  | {
      kind: "user";
      key: string;
      anonymous: false;
      email?: string;
      email_domain?: string;
      role?: string;
      created_at?: string;
      custom: { role?: string };
    };

export interface LDDeviceContext {
  kind: "device";
  key: string;
  anonymous: true;
}

export interface LDMultiContext {
  kind: "multi";
  user: LDUserContext;
  device: LDDeviceContext;
}

export type LDContext = LDUserContext | LDMultiContext;

// Mirror the context built by the backend
// (feature_flag.py:_fetch_user_context_data) so LaunchDarkly targeting
// rules evaluate identically on both sides.
//
// The auth session emits `Z`-suffixed ISO; backend emits `+00:00` — LD date matchers accept both.
//
// `anonymousID` is the first-party anonymous id shared with PostHog (see
// services/analytics/anonymous-id.ts). Logged out, it is the user key, so
// percentage rollouts are stable per visitor instead of identical for every
// visitor. Logged in, it rides along as a `device` context so a rule that
// buckets by device keeps the same arm across signup. Rules on the `user`
// kind are unchanged.
export function buildLDContext(
  user: User | null,
  anonymousID?: string | null,
): LDContext {
  if (!user) {
    return { kind: "user", key: anonymousID || "anonymous", anonymous: true };
  }

  const userContext: LDUserContext = {
    kind: "user",
    key: user.id,
    anonymous: false,
    ...(user.email && {
      email: user.email,
      email_domain: user.email.split("@").at(-1),
    }),
    ...(user.role && { role: user.role }),
    ...(user.created_at && { created_at: user.created_at }),
    custom: {
      ...(user.role && { role: user.role }),
    },
  };

  if (!anonymousID) return userContext;

  return {
    kind: "multi",
    user: userContext,
    device: { kind: "device", key: anonymousID, anonymous: true },
  };
}
