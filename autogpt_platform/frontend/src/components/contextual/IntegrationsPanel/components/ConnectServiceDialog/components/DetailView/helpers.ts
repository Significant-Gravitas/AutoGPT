import type { ProviderTiers } from "@/app/api/__generated__/models/providerTiers";

type ValidationDetailItem = { msg?: unknown };

function readDetail(value: unknown): string | null {
  if (typeof value !== "object" || value === null) return null;
  const detail = (value as { detail?: unknown }).detail;

  if (typeof detail === "string" && detail.length > 0) return detail;

  if (Array.isArray(detail)) {
    const messages = detail
      .map((item: ValidationDetailItem) =>
        typeof item?.msg === "string" ? item.msg : null,
      )
      .filter((msg): msg is string => msg !== null);
    if (messages.length > 0) return messages.join(", ");
  }

  if (typeof detail === "object" && detail !== null) {
    const message = (detail as { message?: unknown }).message;
    if (typeof message === "string" && message.length > 0) {
      const hint = (detail as { hint?: unknown }).hint;
      return typeof hint === "string" && hint.length > 0
        ? `${message} ${hint}`
        : message;
    }
  }

  return null;
}

// Extracts a human-readable message from an error thrown by the API client.
// The mutator builds `ApiError` via `new Error(detail)`, so when the backend
// returns a non-string `detail` (FastAPI 422 array, or a dict), `error.message`
// is coerced to the useless string "[object Object]". Prefer the structured
// `response.detail` and only fall back to `error.message` when it's usable.
export function getOAuthErrorMessage(error: unknown): string {
  if (typeof error === "object" && error !== null) {
    const fromResponse = readDetail((error as { response?: unknown }).response);
    if (fromResponse) return fromResponse;

    const fromError = readDetail(error);
    if (fromError) return fromError;
  }

  if (
    error instanceof Error &&
    error.message &&
    error.message !== "[object Object]"
  ) {
    return error.message;
  }

  return "Something went wrong. Please try again.";
}

/**
 * "5.6 Terra (Balanced) and 5.6 Sol (Advanced)", from the catalog.
 *
 * Empty when the server named nothing, so the sentence falls back to the
 * general one rather than rendering half of a promise.
 */
export function chatgptModelsSentence(
  providers: ProviderTiers[] | undefined,
): string {
  const chatgpt = (providers ?? []).find(
    (provider) => provider.provider_family === "openai",
  );
  const named = (chatgpt?.tiers ?? [])
    .filter((tier) => tier.display_model)
    .map((tier) => `${tier.display_model} (${tier.label})`);
  if (named.length === 0) return "";
  if (named.length === 1) return named[0];
  return `${named.slice(0, -1).join(", ")} and ${named[named.length - 1]}`;
}
