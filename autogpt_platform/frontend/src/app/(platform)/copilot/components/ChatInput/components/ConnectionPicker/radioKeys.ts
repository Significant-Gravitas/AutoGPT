/**
 * Keyboard behaviour for a radio group, which is not the same as a list of
 * buttons that happen to say `role="radio"`.
 *
 * A radio group is one tab stop: Tab moves past the whole group, and the
 * arrow keys move *and select* within it. Without that, a keyboard user tabs
 * through every option one at a time and a screen reader announces a group
 * whose operating instructions do not match how it behaves.
 *
 * Locked options are skipped rather than focused. They exist to explain a
 * barrier, and stopping on something that cannot be chosen is a dead end.
 */

const NEXT_KEYS = new Set(["ArrowRight", "ArrowDown"]);
const PREV_KEYS = new Set(["ArrowLeft", "ArrowUp"]);

export interface RovingOption<T> {
  value: T;
  disabled?: boolean;
}

/** The single tab stop: the selected option, or the first selectable one. */
export function rovingTabIndex<T>(
  options: RovingOption<T>[],
  option: RovingOption<T>,
  selected: T,
): 0 | -1 {
  const selectable = options.filter((candidate) => !candidate.disabled);
  const active =
    selectable.find((candidate) => candidate.value === selected) ??
    selectable[0];
  return active && active.value === option.value ? 0 : -1;
}

/**
 * Returns the option the key moves to, or null when the key is not one this
 * group handles — in which case the caller must not preventDefault, or it
 * would swallow Tab and Escape.
 */
export function nextRovingValue<T>(
  options: RovingOption<T>[],
  current: T,
  key: string,
): T | null {
  const selectable = options.filter((option) => !option.disabled);
  if (selectable.length === 0) return null;

  if (key === "Home") return selectable[0].value;
  if (key === "End") return selectable[selectable.length - 1].value;

  const step = NEXT_KEYS.has(key) ? 1 : PREV_KEYS.has(key) ? -1 : 0;
  if (step === 0) return null;

  const at = selectable.findIndex((option) => option.value === current);
  // An unselected group starts from whichever end the user arrowed toward.
  if (at === -1) {
    return step === 1
      ? selectable[0].value
      : selectable[selectable.length - 1].value;
  }
  const to = (at + step + selectable.length) % selectable.length;
  return selectable[to].value;
}
