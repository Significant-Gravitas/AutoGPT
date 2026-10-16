/**
 * Removing a file takes its focused Remove button with it. Focus moves to
 * the next failed file's Retry, else to another card in the same message,
 * else to the chat's composer, so a keyboard user keeps their place.
 */
export function focusAfterRemoving(card: HTMLElement | null) {
  const list = card?.parentElement;
  if (!card || !list) return;
  const cards = Array.from(list.children).filter(
    (el): el is HTMLElement => el instanceof HTMLElement,
  );
  const index = cards.indexOf(card);
  const others = [
    ...cards.slice(index + 1),
    ...cards.slice(0, index).reverse(),
  ];
  const nextRetry = others
    .map((el) => el.querySelector<HTMLElement>("[data-retry-upload]"))
    .find((button) => button !== null);
  (nextRetry ?? others[0] ?? nearestComposer(list))?.focus();
}

function nearestComposer(from: HTMLElement) {
  for (let el = from.parentElement; el; el = el.parentElement) {
    const composer = el.querySelector<HTMLElement>(
      "[data-slot=input-group-control]",
    );
    if (composer) return composer;
  }
  return null;
}
