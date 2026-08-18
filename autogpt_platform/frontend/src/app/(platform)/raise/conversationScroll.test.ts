import { describe, expect, test } from "vitest";
import {
  MIN_BOTTOM_PADDING,
  bottomPaddingToCenter,
  centerLastChild,
  isPromptStepPair,
  scrollTopToCenterChild,
} from "./conversationScroll";

describe("conversationScroll", () => {
  test("pairs a prompt message with its matching step", () => {
    const prompt = document.createElement("div");
    prompt.id = "autogpt-voice-question";
    const step = document.createElement("div");
    step.id = "voice-step";

    expect(isPromptStepPair(prompt, step)).toBe(true);
    expect(isPromptStepPair(null, step)).toBe(false);

    step.id = "kit-step";
    expect(isPromptStepPair(prompt, step)).toBe(false);
  });

  test("pads enough that a short last item can sit in the vertical center", () => {
    expect(bottomPaddingToCenter(800, 80)).toBe(360);
  });

  test("keeps a minimum bottom padding when the last item is taller than the pane", () => {
    expect(bottomPaddingToCenter(800, 900)).toBe(MIN_BOTTOM_PADDING);
  });

  test("computes scrollTop so the child center matches the pane center", () => {
    expect(
      scrollTopToCenterChild({
        childOffsetTop: 400,
        childHeight: 80,
        viewportHeight: 800,
        maxScrollTop: 1000,
      }),
    ).toBe(40);
  });

  test("aligns a block taller than the pane to its top", () => {
    expect(
      scrollTopToCenterChild({
        childOffsetTop: 1200,
        childHeight: 900,
        viewportHeight: 800,
        maxScrollTop: 3000,
      }),
    ).toBe(1200);
  });

  test("clamps to the scrollable range when there is not enough room", () => {
    expect(
      scrollTopToCenterChild({
        childOffsetTop: 0,
        childHeight: 80,
        viewportHeight: 800,
        maxScrollTop: 20,
      }),
    ).toBe(0);
    expect(
      scrollTopToCenterChild({
        childOffsetTop: 2000,
        childHeight: 80,
        viewportHeight: 800,
        maxScrollTop: 50,
      }),
    ).toBe(50);
  });

  test("centers the last child in the scroll container", () => {
    const container = document.createElement("div");
    Object.assign(container.style, {
      height: "400px",
      overflow: "auto",
      position: "relative",
    });

    const spacer = document.createElement("div");
    spacer.style.height = "1200px";
    container.appendChild(spacer);

    const target = document.createElement("div");
    target.style.height = "80px";
    container.appendChild(target);

    document.body.appendChild(container);

    centerLastChild(container, "auto");

    const childOffsetTop = 1200;
    const expectedTop = scrollTopToCenterChild({
      childOffsetTop,
      childHeight: 80,
      viewportHeight: 400,
      maxScrollTop: container.scrollHeight - container.clientHeight,
    });

    expect(container.scrollTop).toBe(expectedTop);
    expect(
      Number.parseFloat(container.style.paddingBottom),
    ).toBeGreaterThanOrEqual(MIN_BOTTOM_PADDING);

    document.body.removeChild(container);
  });

  test("scrolls less far when centering a prompt+step block than the step alone", () => {
    const pairTop = scrollTopToCenterChild({
      childOffsetTop: 1200,
      childHeight: 600,
      viewportHeight: 400,
      maxScrollTop: 2000,
    });
    const stepOnlyTop = scrollTopToCenterChild({
      childOffsetTop: 1280,
      childHeight: 520,
      viewportHeight: 400,
      maxScrollTop: 2000,
    });

    expect(pairTop).toBeLessThan(stepOnlyTop);
  });
});
