import {
  act,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PanelResizeHandle } from "../PanelResizeHandle";

// The handle hands the panel at most one new width per animation frame, and
// the last one when the pointer comes up, so a drag does not write to the
// store (and re-render every subscriber) on each pointer move.

const PANEL_WIDTH = 400;
const CONTAINER_WIDTH = 1600;
const START_X = 1000;

let frames: Map<number, FrameRequestCallback>;
let nextFrameID: number;

function runFrame() {
  const queued = [...frames.values()];
  frames.clear();
  act(() => queued.forEach((callback) => callback(performance.now())));
}

function renderHandle() {
  const onWidthChange = vi.fn();
  const onResizingChange = vi.fn();
  const view = render(
    <div data-container>
      <div data-panel>
        <PanelResizeHandle
          panelSelector="[data-panel]"
          onWidthChange={onWidthChange}
          onResizingChange={onResizingChange}
          minWidth={200}
          maxWidth={800}
          reservedWidth={0}
        />
      </div>
    </div>,
  );
  const handle = screen.getByRole("separator", { name: "Resize panel" });
  return { ...view, handle, onWidthChange, onResizingChange };
}

function moveTo(clientX: number) {
  fireEvent.pointerMove(document, { clientX });
}

beforeEach(() => {
  frames = new Map();
  nextFrameID = 1;
  vi.spyOn(window, "requestAnimationFrame").mockImplementation((callback) => {
    const id = nextFrameID++;
    frames.set(id, callback);
    return id;
  });
  vi.spyOn(window, "cancelAnimationFrame").mockImplementation((id) => {
    frames.delete(id);
  });
  vi.spyOn(HTMLElement.prototype, "offsetWidth", "get").mockImplementation(
    function (this: HTMLElement) {
      if (this.hasAttribute("data-panel")) return PANEL_WIDTH;
      if (this.hasAttribute("data-container")) return CONTAINER_WIDTH;
      return 0;
    },
  );
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("PanelResizeHandle", () => {
  it("writes one width for ten pointer moves within one frame", () => {
    const { handle, onWidthChange } = renderHandle();

    fireEvent.pointerDown(handle, { clientX: START_X, pointerId: 1 });
    for (let step = 1; step <= 10; step++) moveTo(START_X - step);

    expect(onWidthChange).not.toHaveBeenCalled();
    runFrame();

    expect(onWidthChange).toHaveBeenCalledTimes(1);
    expect(onWidthChange).toHaveBeenCalledWith(PANEL_WIDTH + 10);
  });

  it("writes the latest width once per frame across a longer drag", () => {
    const { handle, onWidthChange } = renderHandle();

    fireEvent.pointerDown(handle, { clientX: START_X, pointerId: 1 });
    moveTo(START_X - 5);
    moveTo(START_X - 20);
    runFrame();
    moveTo(START_X - 30);
    runFrame();
    runFrame();

    expect(onWidthChange.mock.calls).toEqual([
      [PANEL_WIDTH + 20],
      [PANEL_WIDTH + 30],
    ]);
  });

  it("commits the pending width when the pointer comes up, before the drag ends", () => {
    const { handle, onWidthChange, onResizingChange } = renderHandle();

    fireEvent.pointerDown(handle, { clientX: START_X, pointerId: 1 });
    moveTo(START_X - 50);
    onResizingChange.mockClear();
    fireEvent.pointerUp(handle, { clientX: START_X - 50, pointerId: 1 });

    expect(onWidthChange).toHaveBeenCalledTimes(1);
    expect(onWidthChange).toHaveBeenCalledWith(PANEL_WIDTH + 50);
    // The panel still drops its width tween for the final write.
    expect(onWidthChange.mock.invocationCallOrder[0]).toBeLessThan(
      onResizingChange.mock.invocationCallOrder[0],
    );
    expect(onResizingChange).toHaveBeenLastCalledWith(false);

    runFrame();
    moveTo(START_X - 80);
    runFrame();
    expect(onWidthChange).toHaveBeenCalledTimes(1);
  });

  it("commits the pending width on a document-level pointer cancel", () => {
    const { handle, onWidthChange } = renderHandle();

    fireEvent.pointerDown(handle, { clientX: START_X, pointerId: 1 });
    moveTo(START_X + 100);
    fireEvent.pointerCancel(document);

    expect(onWidthChange).toHaveBeenCalledTimes(1);
    expect(onWidthChange).toHaveBeenCalledWith(PANEL_WIDTH - 100);
    expect(frames.size).toBe(0);
  });

  it("commits the pending width when the panel closes mid-drag", () => {
    const { handle, onWidthChange, unmount } = renderHandle();

    fireEvent.pointerDown(handle, { clientX: START_X, pointerId: 1 });
    moveTo(START_X - 40);
    unmount();

    expect(onWidthChange).toHaveBeenCalledTimes(1);
    expect(onWidthChange).toHaveBeenCalledWith(PANEL_WIDTH + 40);
    expect(frames.size).toBe(0);
  });

  it("still clamps the width to the panel's limits", () => {
    const { handle, onWidthChange } = renderHandle();

    fireEvent.pointerDown(handle, { clientX: START_X, pointerId: 1 });
    moveTo(START_X - 900);
    runFrame();
    moveTo(START_X + 900);
    runFrame();

    expect(onWidthChange.mock.calls).toEqual([[800], [200]]);
  });
});
