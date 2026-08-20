/**
 * The typed reason a provider refused a turn, as it arrives on the stream.
 *
 * Until this existed the client had to sniff substrings out of error text --
 * `"usage limit"`, `"401"` -- to guess what happened, and everything it
 * could not place got the same advice: press Try Again. That advice is
 * wrong for most provider failures. Retrying an expired login, a spent
 * quota or a retired model cannot succeed, and telling someone to retry
 * costs them the time it takes to find out.
 *
 * The server decides `retryable` and `reconnectFixesIt` so a second
 * implementation of that judgement cannot drift from the one that classified
 * the failure.
 */

export type ProviderFailureKind =
  | "auth_expired"
  | "invalid_credential"
  | "usage_limit"
  | "model_unavailable"
  | "policy_denied"
  | "entitlement_required"
  | "transient";

export interface ProviderFailure {
  kind: ProviderFailureKind;
  message: string;
  authProvider: string | null;
  credentialId: string | null;
  resetsAt: number | null;
  retryable: boolean;
  reconnectFixesIt: boolean;
}

const KINDS = new Set<string>([
  "auth_expired",
  "invalid_credential",
  "usage_limit",
  "model_unavailable",
  "policy_denied",
  "entitlement_required",
  "transient",
]);

export function parseProviderFailurePart(dataPart: {
  type: string;
  data?: unknown;
}): ProviderFailure | null {
  if (dataPart.type !== "data-provider-failure") return null;
  const data = dataPart.data as Partial<ProviderFailure> | undefined;
  if (!data || typeof data.kind !== "string" || !KINDS.has(data.kind)) {
    return null;
  }
  return {
    kind: data.kind as ProviderFailureKind,
    message: typeof data.message === "string" ? data.message : "",
    authProvider: data.authProvider ?? null,
    credentialId: data.credentialId ?? null,
    resetsAt: typeof data.resetsAt === "number" ? data.resetsAt : null,
    retryable: data.retryable === true,
    reconnectFixesIt: data.reconnectFixesIt === true,
  };
}

interface FailureCopy {
  title: string;
  description: string;
}

const COPY_BY_KIND: Record<ProviderFailureKind, FailureCopy> = {
  auth_expired: {
    title: "Your connection needs signing in again",
    description: "Reconnect the account in Settings, then send this again.",
  },
  invalid_credential: {
    title: "That connection can't be used",
    description: "Reconnect the account in Settings, then send this again.",
  },
  usage_limit: {
    title: "You've hit this connection's limit",
    description: "Switch connection to keep going, or wait for it to reset.",
  },
  model_unavailable: {
    title: "That model isn't available",
    description: "Pick another model tier, or switch connection.",
  },
  policy_denied: {
    title: "The provider declined this request",
    description: "Rewording it may help. Retrying it unchanged won't.",
  },
  entitlement_required: {
    title: "Your plan doesn't include this connection",
    description: "Switch connection, or see plans to unlock it.",
  },
  transient: {
    title: "Connection hiccup",
    description: "Something went wrong on the way to the model. Try again.",
  },
};

/**
 * What to tell the user, preferring the provider's own words for the detail.
 *
 * The reset time is appended only when the provider reported one -- an
 * invented "try again in an hour" is worse than silence, because people plan
 * around it.
 */
export function describeProviderFailure(failure: ProviderFailure): FailureCopy {
  const base = COPY_BY_KIND[failure.kind];
  const reset = formatResetHint(failure.resetsAt);
  const detail = failure.message.trim() || base.description;
  return {
    title: base.title,
    description: reset ? `${base.description} ${reset}` : detail,
  };
}

function formatResetHint(resetsAt: number | null): string | null {
  if (resetsAt === null) return null;
  const seconds = resetsAt - Math.floor(Date.now() / 1000);
  if (seconds <= 0) return null;
  if (seconds < 90) return "Resets in under a minute.";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `Resets in about ${minutes} minutes.`;
  const hours = Math.round(minutes / 60);
  return hours === 1
    ? "Resets in about an hour."
    : `Resets in about ${hours} hours.`;
}
