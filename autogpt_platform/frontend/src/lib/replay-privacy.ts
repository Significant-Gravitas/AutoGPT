/**
 * Keeps an element and everything inside it out of session replays: PostHog
 * blocks `ph-no-capture` (its default `blockClass`) and Sentry blocks
 * `.sentry-block` (its default block selector). Blocking rather than masking,
 * because the Text and Button atoms opt back in with `sentry-unmask`, which
 * outranks a `sentry-mask` set further up the tree.
 *
 * Used where the screen shows what a connected account returned, such as
 * Gmail or Drive content, which Google's Limited Use policy keeps from staff.
 */
export const HIDDEN_FROM_REPLAY = "ph-no-capture sentry-block";
