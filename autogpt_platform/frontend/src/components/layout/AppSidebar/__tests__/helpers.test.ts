import { afterEach, describe, expect, it, vi, type Mock } from "vitest";

import {
  afterAnimations,
  scrollSidebarTo,
  scrollSidebarToWhenReachable,
} from "../helpers";

function rectAt(top: number) {
  return new DOMRect(0, top, 0, 0);
}

function fakeAnimation(endTime: number) {
  let finish!: () => void;
  let cancel!: () => void;
  const animation = {
    playState: "running",
    effect: { getComputedTiming: () => ({ endTime }) },
  } as unknown as Animation;
  Object.defineProperty(animation, "finished", {
    value: new Promise<Animation>((resolve, reject) => {
      finish = () => {
        Object.assign(animation, { playState: "finished" });
        resolve(animation);
      };
      cancel = () => {
        Object.assign(animation, { playState: "idle" });
        reject(new DOMException("Cancelled", "AbortError"));
      };
    }),
  });
  return { animation, finish, cancel };
}

function elementWithAnimations(...rounds: Animation[][]) {
  const element = document.createElement("div");
  const getAnimations = vi.fn(() => rounds[rounds.length - 1]);
  rounds
    .slice(0, -1)
    .forEach((animations) => getAnimations.mockReturnValueOnce(animations));
  element.getAnimations = getAnimations;
  return element;
}

function flushPromises() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function sidebarScrollArea({
  scrollTop = 0,
  scrollHeight = 2000,
  clientHeight = 800,
} = {}) {
  const container = document.createElement("div");
  container.setAttribute("data-sidebar", "content");
  container.style.scrollPaddingTop = "64px";
  const target = document.createElement("div");
  const list = document.createElement("div");
  container.append(target, list);
  document.body.appendChild(container);
  Object.defineProperty(container, "scrollTop", { value: scrollTop });
  Object.defineProperty(container, "scrollHeight", {
    value: scrollHeight,
    configurable: true,
  });
  Object.defineProperty(container, "clientHeight", { value: clientHeight });
  container.getBoundingClientRect = () => rectAt(50);
  target.getBoundingClientRect = () => rectAt(350);
  const scrollTo = vi.fn();
  container.scrollTo = scrollTo;
  return { container, target, list, scrollTo };
}

afterEach(() => {
  document.body.innerHTML = "";
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("scrollSidebarTo", () => {
  it("scrolls the sidebar so the target sits just below its scroll padding", () => {
    const { target, scrollTo } = sidebarScrollArea({ scrollTop: 100 });

    expect(scrollSidebarTo(target, "smooth")).toBe(true);
    expect(scrollTo).toHaveBeenCalledWith({ top: 336, behavior: "smooth" });
  });

  it("reports when the sidebar cannot scroll that far yet", () => {
    const { target, scrollTo } = sidebarScrollArea({
      scrollHeight: 900,
      clientHeight: 800,
    });

    expect(scrollSidebarTo(target, "auto")).toBe(false);
    expect(scrollTo).toHaveBeenCalledWith({ top: 236, behavior: "auto" });
  });

  it("does nothing when the target is outside the sidebar", () => {
    const target = document.createElement("div");
    const outer = document.createElement("div");
    outer.appendChild(target);
    document.body.appendChild(outer);
    const scrollTo = vi.fn();
    outer.scrollTo = scrollTo;

    scrollSidebarTo(target, "auto");

    expect(scrollTo).not.toHaveBeenCalled();
  });
});

describe("scrollSidebarToWhenReachable", () => {
  function stubResizeObserver() {
    const observers: {
      callback: () => void;
      observe: Mock;
      disconnect: Mock;
    }[] = [];
    vi.stubGlobal(
      "ResizeObserver",
      class {
        callback: () => void;
        disconnect = vi.fn();
        observe = vi.fn();
        constructor(callback: () => void) {
          this.callback = callback;
          observers.push(this);
        }
      },
    );
    return observers;
  }

  it("scrolls once and stops when the target is already reachable", () => {
    const observers = stubResizeObserver();
    const { target, list, scrollTo } = sidebarScrollArea();

    scrollSidebarToWhenReachable(target, list, "smooth");

    expect(scrollTo).toHaveBeenCalledTimes(1);
    expect(observers).toHaveLength(0);
  });

  it("scrolls again as the list grows until the target is reachable", () => {
    const observers = stubResizeObserver();
    const { container, target, list, scrollTo } = sidebarScrollArea({
      scrollHeight: 900,
    });

    scrollSidebarToWhenReachable(target, list, "auto");
    expect(observers).toHaveLength(1);
    expect(observers[0].observe.mock.lastCall?.[0]).toBe(list);
    observers[0].callback();
    expect(observers[0].disconnect).not.toHaveBeenCalled();

    Object.defineProperty(container, "scrollHeight", { value: 2000 });
    observers[0].callback();

    expect(scrollTo).toHaveBeenCalledTimes(3);
    expect(scrollTo).toHaveBeenLastCalledWith({ top: 236, behavior: "auto" });
    expect(observers[0].disconnect).toHaveBeenCalled();
  });

  it.each(["wheel", "touchstart", "pointerdown", "keydown"])(
    "stops following on %s in the sidebar, not on its own scrolling or input elsewhere",
    (type) => {
      const observers = stubResizeObserver();
      const { container, target, list } = sidebarScrollArea({
        scrollHeight: 900,
      });
      scrollSidebarToWhenReachable(target, list, "smooth");

      container.dispatchEvent(new Event("scroll"));
      document.body.dispatchEvent(new Event(type, { bubbles: true }));
      expect(observers[0].disconnect).not.toHaveBeenCalled();

      target.dispatchEvent(new Event(type, { bubbles: true }));
      expect(observers[0].disconnect).toHaveBeenCalled();
    },
  );

  it("gives up after a few seconds, or when stopped", () => {
    vi.useFakeTimers();
    const observers = stubResizeObserver();
    const short = sidebarScrollArea({ scrollHeight: 900 });
    scrollSidebarToWhenReachable(short.target, short.list, "auto");
    const other = sidebarScrollArea({ scrollHeight: 900 });
    const stop = scrollSidebarToWhenReachable(other.target, other.list, "auto");

    stop();
    expect(observers[1].disconnect).toHaveBeenCalled();
    vi.advanceTimersByTime(4999);
    expect(observers[0].disconnect).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(observers[0].disconnect).toHaveBeenCalled();
  });
});

describe("afterAnimations", () => {
  it("waits for every finite animation and ignores endless ones", async () => {
    const short = fakeAnimation(150);
    const long = fakeAnimation(260);
    const spinner = fakeAnimation(Infinity);
    const element = elementWithAnimations([
      short.animation,
      long.animation,
      spinner.animation,
    ]);
    const callback = vi.fn();

    afterAnimations(element, callback);
    expect(element.getAnimations).toHaveBeenCalledWith({ subtree: true });
    short.finish();
    await flushPromises();
    expect(callback).not.toHaveBeenCalled();
    long.finish();

    await vi.waitFor(() => expect(callback).toHaveBeenCalledTimes(1));
  });

  it("waits for an animation that replaces a cancelled one", async () => {
    const first = fakeAnimation(260);
    const replacement = fakeAnimation(260);
    const callback = vi.fn();

    afterAnimations(
      elementWithAnimations([first.animation], [replacement.animation]),
      callback,
    );
    first.cancel();
    await flushPromises();
    expect(callback).not.toHaveBeenCalled();
    replacement.finish();

    await vi.waitFor(() => expect(callback).toHaveBeenCalledTimes(1));
  });

  it("calls back right away when nothing is animating", async () => {
    const callback = vi.fn();

    afterAnimations(elementWithAnimations([]), callback);

    await vi.waitFor(() => expect(callback).toHaveBeenCalledTimes(1));
  });

  it("stops waiting after a few rounds of new animations", async () => {
    const element = document.createElement("div");
    element.getAnimations = vi.fn(() => {
      const animation = fakeAnimation(150);
      animation.finish();
      Object.assign(animation.animation, { playState: "running" });
      return [animation.animation];
    });
    const callback = vi.fn();

    afterAnimations(element, callback);

    await vi.waitFor(() => expect(callback).toHaveBeenCalledTimes(1));
    expect(element.getAnimations).toHaveBeenCalledTimes(5);
  });

  it("never calls back once cancelled", async () => {
    const opening = fakeAnimation(260);
    const callback = vi.fn();

    const cancel = afterAnimations(
      elementWithAnimations([opening.animation]),
      callback,
    );
    cancel();
    opening.finish();
    await flushPromises();

    expect(callback).not.toHaveBeenCalled();
  });
});
