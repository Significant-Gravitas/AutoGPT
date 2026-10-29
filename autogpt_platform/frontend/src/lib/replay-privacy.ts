/**
 * Keeps an element and everything inside it out of session replays: PostHog
 * blocks `ph-no-capture` (its default `blockClass`).
 *
 * Used where the screen shows what a connected account returned, such as
 * Gmail or Drive content, which Google's Limited Use policy keeps from staff.
 */
export const HIDDEN_FROM_REPLAY = "ph-no-capture";
