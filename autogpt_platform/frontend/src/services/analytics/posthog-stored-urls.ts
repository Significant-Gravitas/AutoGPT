import posthog, { type CaptureResult, type PostHog } from "posthog-js";
import { stripCredentialsFromValue } from "./url-credentials";

/**
 * Masks the credentials in the URLs posthog-js keeps in its own storage. It
 * sends the first page it stored with every /flags/ request, which skips
 * before_send, and keeps it in the `.agpt.co` cookie that agpt.co's PostHog
 * reads too. Every stored key is walked, so a key a posthog-js upgrade renames
 * is still covered.
 */
export function stripStoredURLCredentials(client: PostHog = posthog) {
  const stored = client.persistence?.props;
  if (!stored) return;
  const changed: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(stored)) {
    const stripped = stripCredentialsFromValue(value);
    if (JSON.stringify(stripped) !== JSON.stringify(value)) {
      changed[key] = stripped;
    }
  }
  if (Object.keys(changed).length) client.register(changed);
}

/**
 * PostHog `before_send` hook: posthog-js stores the first page just before
 * before_send runs, and the next /flags/ request reads it from storage.
 */
export function keepStoredURLsStripped(
  event: CaptureResult | null,
): CaptureResult | null {
  stripStoredURLCredentials();
  return event;
}
