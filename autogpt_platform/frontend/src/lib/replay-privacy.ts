/**
 * Keeps an element and everything inside it out of session replays: PostHog
 * blocks `ph-no-capture` (its default `blockClass`) and Sentry blocks
 * `.sentry-block` (its default block selector). Blocking rather than masking,
 * because Sentry applies the nearest marker: an element that is
 * `UNMASKED_IN_REPLAY` inside a `sentry-mask` shows its text, and a
 * `sentry-mask` nearer to the text than the unmask masks it again.
 *
 * Used where the screen shows what a connected account returned, such as
 * Gmail or Drive content, which Google's Limited Use policy keeps from staff.
 */
export const HIDDEN_FROM_REPLAY = "ph-no-capture sentry-block";

/**
 * Sentry replays mask all text. This class shows an element's text, and its
 * own text attributes (`aria-label`, `title`, `placeholder`...), in the replay.
 *
 * Only for fixed UI copy that is the same for every user: a nav label, a menu
 * item, a static heading. Put it on the element that holds that string, never
 * on a wrapper around anything else. Sentry applies the nearest marker, so an
 * unmasked element shows the text of everything inside it unless an element
 * nearer to that text is `MASKED_IN_REPLAY`.
 *
 * ESLint enforces it: an unmasked element (this class or the `unmask` prop of
 * `Text` and `Button`) may only hold string literals, `fixedCopy(...)` and
 * icons.
 */
export const UNMASKED_IN_REPLAY = "sentry-unmask";

/**
 * Keeps user content masked even inside an element that is
 * `UNMASKED_IN_REPLAY`, because the nearer marker wins.
 */
export const MASKED_IN_REPLAY = "sentry-mask";

declare const fixedCopyBrand: unique symbol;

/** A string written in our code, the same for every user. */
export type FixedCopy = string & { readonly [fixedCopyBrand]: true };

// A string literal type, a union of them, or copy already checked. A string
// built at runtime (`string`, or a template such as `Hi ${string}`) is not.
type FixedCopyInput<T extends string> = T extends FixedCopy
  ? T
  : Record<never, never> extends Record<T, 1>
    ? never
    : T;

/**
 * Marks fixed UI copy so an unmasked element may show it. The compiler only
 * accepts a string literal (or a choice between literals) or a value that is
 * already `FixedCopy`, so a user's name or anything interpolated can't pass.
 */
export function fixedCopy<T extends string>(
  text: FixedCopyInput<T>,
): FixedCopy {
  return text as unknown as FixedCopy;
}

/**
 * Attributes Sentry masks unless the element itself is unmasked. Sentry's
 * default covers only `title`, `placeholder` and `aria-label`; the rest is
 * recorded as written, and these carry user content: amounts and usage in
 * `aria-valuetext` and `aria-valuenow`, names in `data-text` and
 * `data-placeholder`, file names in `download`, option labels and values.
 */
export const MASKED_ATTRIBUTES = [
  "title",
  "placeholder",
  "aria-label",
  "aria-description",
  "aria-placeholder",
  "aria-roledescription",
  "aria-valuenow",
  "aria-valuetext",
  "alt",
  "label",
  "value",
  "download",
  "data-text",
  "data-placeholder",
  "data-title",
  "data-value",
  "data-skill-name",
];

/**
 * Text is masked unless an element opts in with `UNMASKED_IN_REPLAY`, and so
 * are `MASKED_ATTRIBUTES`. Other attributes, such as ids, `data-testid` and
 * `role`, are recorded as written. So are `href` and `src`: Sentry passes no
 * replay DOM event through a hook we could redact them in, so a link to a
 * creator's or a workflow's page shows its address.
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
  maskAttributes: MASKED_ATTRIBUTES,
  mask: ["input, textarea, select, [contenteditable]"],
  unmask: [".sentry-unmask, [data-sentry-unmask]"],
};
