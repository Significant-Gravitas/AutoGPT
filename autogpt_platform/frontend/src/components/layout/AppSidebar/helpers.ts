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

export function whenHeightSettles(element: HTMLElement, callback: () => void) {
  let lastHeight = -1;
  let frame = requestAnimationFrame(check);

  function check() {
    const height = element.getBoundingClientRect().height;
    if (height === lastHeight) {
      callback();
      return;
    }
    lastHeight = height;
    frame = requestAnimationFrame(check);
  }

  return () => cancelAnimationFrame(frame);
}
