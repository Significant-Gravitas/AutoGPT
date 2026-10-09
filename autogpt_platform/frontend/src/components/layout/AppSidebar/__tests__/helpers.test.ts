import { afterEach, describe, expect, it, vi } from "vitest";

import { afterAnimations, scrollSidebarTo } from "../helpers";

function rectAt(top: number) {
  return new DOMRect(0, top, 0, 0);
}

function fakeAnimation(endTime: number) {
  let finish!: () => void;
  let cancel!: () => void;
  const animation = {
    effect: { getComputedTiming: () => ({ endTime }) },
  } as unknown as Animation;
  Object.defineProperty(animation, "finished", {
    value: new Promise<Animation>((resolve, reject) => {
      finish = () => resolve(animation);
      cancel = () => reject(new DOMException("Cancelled", "AbortError"));
    }),
  });
  return { animation, finish, cancel };
}

function elementWithAnimations(animations: Animation[]) {
  const element = document.createElement("div");
  element.getAnimations = vi.fn(() => animations);
  return element;
}

afterEach(() => {
  document.body.innerHTML = "";
});

describe("scrollSidebarTo", () => {
  it("scrolls the sidebar so the target sits just below its scroll padding", () => {
    const container = document.createElement("div");
    container.setAttribute("data-sidebar", "content");
    container.style.scrollPaddingTop = "64px";
    const target = document.createElement("div");
    container.appendChild(target);
    document.body.appendChild(container);
    Object.defineProperty(container, "scrollTop", { value: 100 });
    container.getBoundingClientRect = () => rectAt(50);
    target.getBoundingClientRect = () => rectAt(350);
    const scrollTo = vi.fn();
    container.scrollTo = scrollTo;

    scrollSidebarTo(target, "smooth");

    expect(scrollTo).toHaveBeenCalledWith({ top: 336, behavior: "smooth" });
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
    await Promise.resolve();
    expect(callback).not.toHaveBeenCalled();
    long.finish();

    await vi.waitFor(() => expect(callback).toHaveBeenCalledTimes(1));
  });

  it("still calls back when an animation is cancelled", async () => {
    const opening = fakeAnimation(260);
    const callback = vi.fn();

    afterAnimations(elementWithAnimations([opening.animation]), callback);
    opening.cancel();

    await vi.waitFor(() => expect(callback).toHaveBeenCalledTimes(1));
  });

  it("calls back right away when nothing is animating", async () => {
    const callback = vi.fn();

    afterAnimations(elementWithAnimations([]), callback);

    await vi.waitFor(() => expect(callback).toHaveBeenCalledTimes(1));
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
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(callback).not.toHaveBeenCalled();
  });
});
