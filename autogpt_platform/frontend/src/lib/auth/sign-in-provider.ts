import type { GenericEndpointContext } from "@better-auth/core";

// A sign-in finished with an emailed code runs in the code's own endpoint
// (sign-in-code-plugin.ts), which records its provider here before it links
// anything, so the code hook (require-sign-in-code.ts) lets that link through.
// Keyed by the endpoint context: Better Auth hands `validateUserInfo` the same
// object the endpoint runs with (`runWithEndpointContext`). If a release ever
// stops doing that, the code-link tests in sign-in-code-flow.test.ts fail.
const confirmedByCode = new WeakMap<object, string>();

export function markConfirmedByCode(ctx: object, providerId: string) {
  confirmedByCode.set(ctx, providerId);
}

/** The provider whose emailed code this endpoint context just accepted. */
export function providerConfirmedByCode(ctx: unknown) {
  return typeof ctx === "object" && ctx !== null
    ? confirmedByCode.get(ctx)
    : undefined;
}

interface ProviderEntry {
  id: string;
  options?: { disableSignUp?: boolean };
  accountSubject?: (input: {
    tokens: object;
    profile: Record<string, unknown>;
  }) => unknown;
}

/** The configured social provider with this id, as Better Auth holds it. */
export async function findProvider(
  ctx: GenericEndpointContext,
  providerId: string,
) {
  for (const entry of ctx.context.socialProviders as unknown[]) {
    const provider = (
      typeof entry === "function" ? await entry() : entry
    ) as ProviderEntry;
    if (provider?.id === providerId) return provider;
  }
  return null;
}

// The account key Better Auth gives the provider's account: the same
// `accountSubject` its callback uses, so finishing with the code links exactly
// the account the provider signed in.
export async function accountIDOf(
  ctx: GenericEndpointContext,
  providerId: string,
  profile: Record<string, unknown> | undefined,
) {
  const provider = await findProvider(ctx, providerId);
  if (!profile || !provider?.accountSubject) return null;
  try {
    const subject = await provider.accountSubject({ tokens: {}, profile });
    const id =
      typeof subject === "string" || typeof subject === "number"
        ? String(subject)
        : "";
    return id.trim() ? id : null;
  } catch {
    return null;
  }
}
