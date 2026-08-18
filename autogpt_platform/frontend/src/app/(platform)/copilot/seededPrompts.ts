/**
 * Fixed registry of URL-seedable first prompts (`/copilot?seed=<key>`).
 *
 * Keys, never free text: an arbitrary-prompt query param would let any link
 * inject a first message into the user's chat. Unknown keys are ignored and
 * cleared by the page.
 */
export const SEEDED_PROMPTS = {
  "memory-summary": "Give me a summary of everything you know about me",
  "memory-forget":
    "I'd like you to forget something from your memory. Ask me what to " +
    "forget, search your memory for matching facts, show me what you find, " +
    "and only delete after I confirm.",
} as const;

export type SeededPromptKey = keyof typeof SEEDED_PROMPTS;

export function getSeededPrompt(key: string | null): string | null {
  if (!key) return null;
  if (!(key in SEEDED_PROMPTS)) return null;
  return SEEDED_PROMPTS[key as SeededPromptKey];
}

export function buildSeededChatHref(
  key: SeededPromptKey,
  expertId?: string | null,
): string {
  const params = new URLSearchParams({ seed: key });
  if (expertId) params.set("expertId", expertId);
  return `/copilot?${params.toString()}`;
}
