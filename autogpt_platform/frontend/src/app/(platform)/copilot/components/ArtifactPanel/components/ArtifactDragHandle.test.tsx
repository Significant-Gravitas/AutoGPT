import { fireEvent, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ArtifactDragHandle } from "./ArtifactDragHandle";

function renderHandle(onWidthChange = vi.fn(), panelWidth = 600) {
  const utils = render(
    <div
      data-artifact-panel
      style={{
        width: `${panelWidth}px`,
        height: "400px",
        position: "relative",
      }}
    >
      <ArtifactDragHandle onWidthChange={onWidthChange} />
    </div>,
  );
  const panel = utils.container.querySelector(
    "[data-artifact-panel]",
  ) as HTMLElement;
  // happy-dom doesn't compute layout; stub offsetWidth so the handle reads
  // the intended starting width.
  Object.defineProperty(panel, "offsetWidth", {
    value: panelWidth,
    configurable: true,
  });
  const handle = utils.container.querySelector(
    '[role="separator"]',
  ) as HTMLElement;
  return { handle, onWidthChange, ...utils };
}

// jsdom/happy-dom don't implement pointer capture by default — stub them so
// the component can still exercise the capture calls.
function installPointerCaptureStub() {
  const setPointerCapture = vi.fn();
  const releasePointerCapture = vi.fn();
  (
    HTMLElement.prototype as unknown as {
      setPointerCapture: typeof setPointerCapture;
    }
  ).setPointerCapture = setPointerCapture;
  (
    HTMLElement.prototype as unknown as {
      releasePointerCapture: typeof releasePointerCapture;
    }
  ).releasePointerCapture = releasePointerCapture;
  return { setPointerCapture, releasePointerCapture };
}

describe("ArtifactDragHandle", () => {
  let spies: ReturnType<typeof installPointerCaptureStub>;

  beforeEach(() => {
    spies = installPointerCaptureStub();
    Object.defineProperty(window, "innerWidth", {
      value: 1200,
      writable: true,
      configurable: true,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  // SECRT-2256: when the cursor drifts over a sandboxed iframe mid-drag, the
  // iframe eats pointermove/pointerup and the drag gets stuck. setPointerCapture
  // routes all subsequent pointer events to the handle regardless of what's
  // under the cursor, which fixes both "can't drag right" and "drag doesn't
  // stop on release".
  it("captures the pointer on pointerdown so drags survive the cursor drifting over iframes (SECRT-2256)", () => {
    const { handle } = renderHandle();

    fireEvent.pointerDown(handle, { clientX: 500, pointerId: 7 });

    expect(spies.setPointerCapture).toHaveBeenCalledWith(7);
  });

  it("releases the pointer capture when the drag ends", () => {
    const { handle } = renderHandle();

    fireEvent.pointerDown(handle, { clientX: 500, pointerId: 7 });
    fireEvent.pointerUp(handle, { clientX: 400, pointerId: 7 });

    expect(spies.releasePointerCapture).toHaveBeenCalledWith(7);
  });

  it("calls onWidthChange with the expanded width when dragging leftwards", () => {
    const onWidthChange = vi.fn();
    const { handle } = renderHandle(onWidthChange);

    fireEvent.pointerDown(handle, { clientX: 800, pointerId: 1 });
    fireEvent.pointerMove(document, { clientX: 700, pointerId: 1 });

    // startWidth is 600 (container), delta = 800 - 700 = 100 → newWidth 700
    expect(onWidthChange).toHaveBeenCalledWith(700);
  });

  it("calls onWidthChange with the shrunk width when dragging rightwards", () => {
    const onWidthChange = vi.fn();
    const { handle } = renderHandle(onWidthChange);

    fireEvent.pointerDown(handle, { clientX: 800, pointerId: 1 });
    fireEvent.pointerMove(document, { clientX: 900, pointerId: 1 });

    // delta = -100 → newWidth 500
    expect(onWidthChange).toHaveBeenCalledWith(500);
  });

  it("clamps to minWidth and maxWidth", () => {
    const onWidthChange = vi.fn();
    const { handle } = renderHandle(onWidthChange);

    fireEvent.pointerDown(handle, { clientX: 800, pointerId: 1 });

    // Drag way left → want huge width, should clamp at 85% of 1200 = 1020
    fireEvent.pointerMove(document, { clientX: -5000, pointerId: 1 });
    expect(onWidthChange).toHaveBeenLastCalledWith(1020);

    // Drag way right → want tiny width, should clamp at minWidth 320
    fireEvent.pointerMove(document, { clientX: 5000, pointerId: 1 });
    expect(onWidthChange).toHaveBeenLastCalledWith(320);
  });

  it("stops dragging on pointerup so subsequent cursor moves don't resize", () => {
    const onWidthChange = vi.fn();
    const { handle } = renderHandle(onWidthChange);

    fireEvent.pointerDown(handle, { clientX: 800, pointerId: 1 });
    fireEvent.pointerUp(handle, { clientX: 800, pointerId: 1 });
    onWidthChange.mockClear();

    fireEvent.pointerMove(document, { clientX: 500, pointerId: 1 });
    expect(onWidthChange).not.toHaveBeenCalled();
  });
});
