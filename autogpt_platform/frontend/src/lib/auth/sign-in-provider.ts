// Which provider a sign-in or sign-up came through, for the checks that
// depend on it. Better Auth hands database hooks the endpoint context that is
// creating the row: `/callback/:id` names the provider in its params, and
// `/sign-in/social` (ID tokens) in its body. A sign-in finished with an
// emailed code runs in the code's own endpoint (sign-in-code-plugin.ts),
// which records the provider here before it creates anything.
const confirmedByCode = new WeakMap<object, string>();

interface EndpointContext {
  params?: { id?: string };
  body?: unknown;
}

export function markConfirmedByCode(ctx: object, providerId: string) {
  confirmedByCode.set(ctx, providerId);
}

/** The provider whose emailed code this endpoint context just accepted. */
export function providerConfirmedByCode(ctx: unknown) {
  return typeof ctx === "object" && ctx !== null
    ? confirmedByCode.get(ctx)
    : undefined;
}

export function signInProviderOf(ctx: EndpointContext | null | undefined) {
  const body = ctx?.body;
  const named =
    typeof body === "object" && body !== null && "provider" in body
      ? body.provider
      : undefined;
  return (
    providerConfirmedByCode(ctx) ??
    ctx?.params?.id ??
    (typeof named === "string" ? named : undefined)
  );
}
