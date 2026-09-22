import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { isTextEntry, useConversationScroll } from "./useConversationScroll";

function Harness() {
  const { scrollRef } = useConversationScroll();

  return (
    <div ref={scrollRef} data-testid="log">
      <div id="autogpt-about-question">Anything about your expert?</div>
      <div id="about-step">
        <textarea data-testid="about" />
      </div>
    </div>
  );
}

function renderHarness() {
  render(<Harness />);
  const log = screen.getByTestId("log");
  // happy-dom has no layout, so the column has to be told it overflows for
  // "did the reader scroll away from the end" to mean anything.
  Object.defineProperty(log, "clientHeight", {
    value: 400,
    configurable: true,
  });
  Object.defineProperty(log, "scrollHeight", {
    value: 1200,
    configurable: true,
  });
  const scrollTo = vi.fn();
  log.scrollTo = scrollTo;

  return { log, scrollTo };
}

async function appendStep(log: HTMLElement) {
  const step = document.createElement("div");
  step.id = "voice-step";
  log.appendChild(step);
}

describe("useConversationScroll", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  test("keeps following when the reader types a space in an answer field", async () => {
    const { log, scrollTo } = renderHarness();

    fireEvent.keyDown(screen.getByTestId("about"), { key: " " });
    await appendStep(log);

    await waitFor(() => expect(scrollTo).toHaveBeenCalled());
  });

  test("stops following when the reader presses a scroll key on the column", async () => {
    const { log, scrollTo } = renderHarness();

    fireEvent.keyDown(log, { key: "PageUp" });
    await appendStep(log);

    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(scrollTo).not.toHaveBeenCalled();
  });

  test("follows again once the reader scrolls back to the end", async () => {
    const { log, scrollTo } = renderHarness();

    fireEvent.keyDown(log, { key: "PageUp" });
    log.scrollTop = 800;
    fireEvent.scroll(log);
    await appendStep(log);

    await waitFor(() => expect(scrollTo).toHaveBeenCalled());
  });
});

describe("isTextEntry", () => {
  test("recognises the fields an answer gets typed into", () => {
    expect(isTextEntry(document.createElement("textarea"))).toBe(true);
    expect(isTextEntry(document.createElement("input"))).toBe(true);
    expect(isTextEntry(document.createElement("select"))).toBe(true);
  });

  test("does not treat the conversation column as a field", () => {
    expect(isTextEntry(document.createElement("div"))).toBe(false);
    expect(isTextEntry(document.createElement("button"))).toBe(false);
    expect(isTextEntry(null)).toBe(false);
  });
});
