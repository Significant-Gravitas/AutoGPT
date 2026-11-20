import { act, cleanup, render } from "@/tests/integrations/test-utils";
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TailSpacer } from "../TailSpacer";

const scroller = { clientHeight: 900 };
// The spacer calls the context once per render.
const spacerRenders = vi.hoisted(() => ({ count: 0 }));
vi.mock("use-stick-to-bottom", () => ({
  useStickToBottomContext: () => {
    spacerRenders.count += 1;
    return { scrollRef: { current: scroller } };
  },
}));

let resize: () => void = () => {};
class MockResizeObserver {
  constructor(callback: () => void) {
    resize = callback;
  }
  observe() {}
  disconnect() {}
  unobserve() {}
}

// Where the newest turn starts and where the spacer sits below it.
const layout = { messageTop: 0, spacerTop: 100 };

function rect(top: number) {
  return { top } as DOMRect;
}

let rerenderParent: () => void = () => {};
let remountRow: () => void = () => {};

function Transcript() {
  const [, setTick] = useState(0);
  const [rowKey, setRowKey] = useState(0);
  rerenderParent = () => setTick((tick) => tick + 1);
  remountRow = () => setRowKey((key) => key + 1);
  return (
    <div>
      <div
        key={rowKey}
        data-message-id="u1"
        ref={stubTop(() => layout.messageTop)}
      />
      <TailSpacer messageID="u1" bottomInset={0} />
    </div>
  );
}

function stubTop(top: () => number) {
  return (el: HTMLElement | null) => {
    if (el) el.getBoundingClientRect = () => rect(top());
  };
}

function renderTranscript() {
  const { container } = render(<Transcript />);
  const spacer = container.querySelector<HTMLElement>("[aria-hidden]")!;
  spacer.getBoundingClientRect = () => rect(layout.spacerTop);
  // The stub lands after the first measure, so measure again with it.
  act(() => resize());
  return spacer;
}

describe("TailSpacer", () => {
  beforeEach(() => {
    spacerRenders.count = 0;
    layout.messageTop = 0;
    layout.spacerTop = 100;
    vi.stubGlobal("ResizeObserver", MockResizeObserver);
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("parks a short turn mid-screen and collapses once the reply fills it", () => {
    const spacer = renderTranscript();
    // (900 - 24) * 0.5 - 100
    expect(spacer.style.height).toBe("338px");

    layout.spacerTop = 600;
    act(() => resize());

    expect(spacer.style.height).toBe("0px");
  });

  it("measures the row's new node after the transcript remounts it", () => {
    const spacer = renderTranscript();
    expect(spacer.style.height).toBe("338px");

    // A windowed transcript drops the row when it scrolls far away and
    // mounts a fresh node when it comes back.
    act(() => remountRow());
    layout.messageTop = 40;
    act(() => resize());

    // (900 - 24) * 0.5 - (100 - 40)
    expect(spacer.style.height).toBe("378px");
  });

  it("does not render when its parent renders with the same props", () => {
    renderTranscript();
    const before = spacerRenders.count;

    act(() => rerenderParent());

    expect(spacerRenders.count).toBe(before);
  });

  it("does not render for a resize that leaves its height unchanged", () => {
    renderTranscript();
    const before = spacerRenders.count;

    act(() => resize());
    act(() => resize());

    expect(spacerRenders.count).toBe(before);
  });
});
