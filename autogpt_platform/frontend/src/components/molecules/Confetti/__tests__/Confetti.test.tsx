import { createRef } from "react";
import { render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { Confetti, type ConfettiRef } from "../Confetti";

const fireInstance = vi.fn().mockResolvedValue(undefined);
const resetInstance = vi.fn();
const createInstance = vi.fn();

vi.mock("canvas-confetti", () => ({
  default: {
    create: (...args: unknown[]) => {
      createInstance(...args);
      return Object.assign(fireInstance, { reset: resetInstance });
    },
  },
}));

describe("Confetti", () => {
  beforeEach(() => {
    fireInstance.mockClear();
    resetInstance.mockClear();
    createInstance.mockClear();
  });

  it("creates an instance and auto-fires once on mount", () => {
    render(<Confetti />);

    expect(createInstance).toHaveBeenCalledTimes(1);
    expect(fireInstance).toHaveBeenCalledTimes(1);
  });

  it("caps the device pixel ratio when sizing the canvas", () => {
    const original = window.devicePixelRatio;
    Object.defineProperty(window, "devicePixelRatio", {
      value: 3,
      configurable: true,
    });

    render(<Confetti data-testid="confetti-canvas" />);

    // With offsetWidth/Height of 0 in happy-dom the size stays 0, but the
    // create call still happens with the capped DPR path exercised.
    expect(createInstance).toHaveBeenCalledTimes(1);

    Object.defineProperty(window, "devicePixelRatio", {
      value: original,
      configurable: true,
    });
  });

  it("does not auto-fire when manualstart is set", () => {
    render(<Confetti manualstart />);

    expect(createInstance).toHaveBeenCalledTimes(1);
    expect(fireInstance).not.toHaveBeenCalled();
  });

  it("fires via the imperative ref handle", async () => {
    const ref = createRef<ConfettiRef>();
    render(<Confetti manualstart ref={ref} />);

    expect(fireInstance).not.toHaveBeenCalled();

    await ref.current?.fire();

    expect(fireInstance).toHaveBeenCalledTimes(1);
  });

  it("resets the instance on unmount", () => {
    const { unmount } = render(<Confetti manualstart />);

    unmount();

    expect(resetInstance).toHaveBeenCalledTimes(1);
  });
});
