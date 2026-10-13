const MAX_ANIMATION_ROUNDS = 5;
const FOLLOW_GROWTH_MS = 5000;
const USER_SCROLL_EVENTS = ["wheel", "touchstart", "pointerdown", "keydown"];

export function scrollSidebarTo(target: HTMLElement, behavior: ScrollBehavior) {
  const container = target.closest<HTMLElement>('[data-sidebar="content"]');
  if (!container) return true;
  const padding = parseFloat(getComputedStyle(container).scrollPaddingTop) || 0;
  const top =
    container.scrollTop +
    target.getBoundingClientRect().top -
    container.getBoundingClientRect().top -
    padding;
  container.scrollTo({ top, behavior });
  return top <= container.scrollHeight - container.clientHeight + 1;
}

export function scrollSidebarToWhenReachable(
  target: HTMLElement,
  growing: Element,
  behavior: ScrollBehavior,
) {
  if (scrollSidebarTo(target, behavior)) return () => {};
  const container = target.closest('[data-sidebar="content"]');
  const observer = new ResizeObserver(() => {
    if (scrollSidebarTo(target, behavior)) stop();
  });
  const timeout = setTimeout(stop, FOLLOW_GROWTH_MS);
  observer.observe(growing);
  for (const type of USER_SCROLL_EVENTS) {
    container?.addEventListener(type, stop, { passive: true });
  }

  function stop() {
    clearTimeout(timeout);
    observer.disconnect();
    for (const type of USER_SCROLL_EVENTS) {
      container?.removeEventListener(type, stop);
    }
  }

  return stop;
}

export function afterAnimations(element: Element, callback: () => void) {
  let isCancelled = false;

  // An animation can be cancelled and restarted while we wait (Radix does
  // this when it measures a reopened collapsible), so check again each round.
  function waitForRunningAnimations(round: number) {
    const running = element
      .getAnimations({ subtree: true })
      .filter(
        (animation) =>
          animation.playState !== "finished" &&
          animation.effect?.getComputedTiming().endTime !== Infinity,
      );
    Promise.allSettled(running.map((animation) => animation.finished)).then(
      () => {
        if (isCancelled) return;
        if (running.length > 0 && round < MAX_ANIMATION_ROUNDS) {
          waitForRunningAnimations(round + 1);
          return;
        }
        callback();
      },
    );
  }

  waitForRunningAnimations(1);
  return () => {
    isCancelled = true;
  };
}
