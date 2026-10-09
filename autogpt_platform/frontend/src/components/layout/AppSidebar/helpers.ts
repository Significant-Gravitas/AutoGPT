export function scrollSidebarTo(target: HTMLElement, behavior: ScrollBehavior) {
  const container = target.closest<HTMLElement>('[data-sidebar="content"]');
  if (!container) return;
  const padding = parseFloat(getComputedStyle(container).scrollPaddingTop) || 0;
  const top =
    container.scrollTop +
    target.getBoundingClientRect().top -
    container.getBoundingClientRect().top -
    padding;
  container.scrollTo({ top, behavior });
}

export function afterAnimations(element: Element, callback: () => void) {
  let isCancelled = false;
  const finiteAnimations = element
    .getAnimations({ subtree: true })
    .filter(
      (animation) => animation.effect?.getComputedTiming().endTime !== Infinity,
    );
  Promise.allSettled(
    finiteAnimations.map((animation) => animation.finished),
  ).then(() => {
    if (!isCancelled) callback();
  });
  return () => {
    isCancelled = true;
  };
}
