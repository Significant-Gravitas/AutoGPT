/**
 * Keeps an element and everything inside it out of session replays: PostHog
 * blocks `ph-no-capture` (its default `blockClass`) and Sentry blocks
 * `.sentry-block` (its default block selector). Blocking rather than masking,
 * because an element that is `UNMASKED_IN_REPLAY` outranks a `sentry-mask`
 * set further up the tree.
 *
 * Used where the screen shows what a connected account returned, such as
 * Gmail or Drive content, which Google's Limited Use policy keeps from staff.
 */
export const HIDDEN_FROM_REPLAY = "ph-no-capture sentry-block";

/**
 * Sentry replays mask all text. This class shows an element's text, and its
 * own `aria-label`, `title` and `placeholder`, in the replay.
 *
 * Only for fixed UI copy that is the same for every user: a nav label, a menu
 * item, a static heading. Put it on the element that holds that string, never
 * on a wrapper around anything else. Sentry applies the nearest marker, so an
 * unmasked element shows the text of everything inside it unless an element
 * nearer to that text is `MASKED_IN_REPLAY`.
 */
export const UNMASKED_IN_REPLAY = "sentry-unmask";

/**
 * Keeps user content masked even inside an element that is
 * `UNMASKED_IN_REPLAY`, because the nearer marker wins.
 */
export const MASKED_IN_REPLAY = "sentry-mask";

/**
 * Everything on screen is masked unless an element opts in with
 * `UNMASKED_IN_REPLAY`, which only copy that is the same
 * for every user should.
 *
 * Sentry applies the nearest marker, and `maskAllInputs` does not outrank an
 * unmasked ancestor: a field inside one would be recorded as typed. So every
 * form field and editable element is masked itself, which is nearer than any
 * ancestor. A field that is itself unmasked shows only its placeholder.
 */
export const REPLAY_PRIVACY_OPTIONS = {
  maskAllText: true,
  maskAllInputs: true,
  blockAllMedia: true,
  mask: ["input, textarea, select, option, [contenteditable]"],
  unmask: [".sentry-unmask, [data-sentry-unmask]"],
};
