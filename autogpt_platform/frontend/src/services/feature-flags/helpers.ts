import type { User } from "@supabase/supabase-js";

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

// Mirror the context built by the backend
// (feature_flag.py:_fetch_user_context_data) so LaunchDarkly targeting
// rules evaluate identically on both sides.
//
// Supabase JS emits `Z`-suffixed ISO; backend emits `+00:00` — LD date matchers accept both.
export function buildLDContext(user: User | null): LDUserContext {
  if (!user) {
    return { kind: "user", key: "anonymous", anonymous: true };
  }

  return {
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
}
