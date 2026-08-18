export const MIN_BOTTOM_PADDING = 64;

const PROMPT_PREFIX = "autogpt-";
const PROMPT_SUFFIX = "-question";
const STEP_SUFFIX = "-step";

export function bottomPaddingToCenter(
  viewportHeight: number,
  lastChildHeight: number,
  minPadding = MIN_BOTTOM_PADDING,
) {
  return Math.max(minPadding, (viewportHeight - lastChildHeight) / 2);
}

export function scrollTopToCenterChild({
  childOffsetTop,
  childHeight,
  viewportHeight,
  maxScrollTop,
}: {
  childOffsetTop: number;
  childHeight: number;
  viewportHeight: number;
  maxScrollTop: number;
}) {
  // A block taller than the pane cannot be centered without pushing its own
  // question off the top, so it is aligned to the top instead.
  const next =
    childHeight >= viewportHeight
      ? childOffsetTop
      : childOffsetTop + childHeight / 2 - viewportHeight / 2;
  return Math.max(0, Math.min(next, Math.max(0, maxScrollTop)));
}

function beatFromPromptId(id: string) {
  if (!id.startsWith(PROMPT_PREFIX) || !id.endsWith(PROMPT_SUFFIX)) return null;
  return id.slice(PROMPT_PREFIX.length, -PROMPT_SUFFIX.length);
}

function beatFromStepId(id: string) {
  if (!id.endsWith(STEP_SUFFIX)) return null;
  return id.slice(0, -STEP_SUFFIX.length);
}

export function isPromptStepPair(
  prompt: Element | null,
  step: HTMLElement,
): boolean {
  if (!(prompt instanceof HTMLElement)) return false;
  const promptBeat = beatFromPromptId(prompt.id);
  const stepBeat = beatFromStepId(step.id);
  return promptBeat !== null && promptBeat === stepBeat;
}

export function scrollTargetBounds(container: HTMLElement) {
  const last = container.lastElementChild;
  if (!(last instanceof HTMLElement)) return null;

  const prev = last.previousElementSibling;
  const containerRect = container.getBoundingClientRect();
  const scrollTop = container.scrollTop;

  if (isPromptStepPair(prev, last)) {
    const prompt = prev as HTMLElement;
    const promptRect = prompt.getBoundingClientRect();
    const stepRect = last.getBoundingClientRect();
    return {
      offsetTop: promptRect.top - containerRect.top + scrollTop,
      height: stepRect.bottom - promptRect.top,
    };
  }

  const childRect = last.getBoundingClientRect();
  return {
    offsetTop: childRect.top - containerRect.top + scrollTop,
    height: childRect.height,
  };
}

export function padContainerToCenterLastChild(container: HTMLElement) {
  const bounds = scrollTargetBounds(container);
  if (!bounds) return;

  const padding = bottomPaddingToCenter(container.clientHeight, bounds.height);
  if (container.style.paddingBottom !== `${padding}px`) {
    container.style.paddingBottom = `${padding}px`;
  }
}

export function centerLastChild(
  container: HTMLElement,
  behavior: ScrollBehavior,
) {
  padContainerToCenterLastChild(container);

  const bounds = scrollTargetBounds(container);
  if (!bounds) return;

  const maxScrollTop = container.scrollHeight - container.clientHeight;

  container.scrollTo({
    top: scrollTopToCenterChild({
      childOffsetTop: bounds.offsetTop,
      childHeight: bounds.height,
      viewportHeight: container.clientHeight,
      maxScrollTop,
    }),
    behavior,
  });
}
