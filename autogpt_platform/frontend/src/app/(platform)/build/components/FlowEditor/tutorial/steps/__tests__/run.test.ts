import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

vi.mock(
  "@/app/(platform)/build/components/FlowEditor/tutorial/helpers",
  () => ({
    waitForElement: vi.fn(),
    fitViewToScreen: vi.fn(),
    highlightElement: vi.fn(),
    removeAllHighlights: vi.fn(),
  }),
);

// Must import after mocks
import {
  waitForElement,
  fitViewToScreen,
  highlightElement,
  removeAllHighlights,
} from "@/app/(platform)/build/components/FlowEditor/tutorial/helpers";
import { createRunSteps } from "../run";

const NODE_OUTPUT_SELECTOR = '[data-tutorial-id="node-output"]';

type StepLike = {
  id?: string;
  beforeShowPromise?: () => Promise<unknown>;
  when?: { show?: () => void; hide?: () => void };
};

function getStep(id: string): StepLike {
  const tour = { next: vi.fn() };
  const steps = createRunSteps(tour) as StepLike[];
  const step = steps.find((s) => s.id === id);
  if (!step) throw new Error(`step "${id}" not found`);
  return step;
}

describe("createRunSteps – show-output step", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("waits for the animated fitView to settle before resolving", async () => {
    vi.mocked(waitForElement).mockResolvedValue(document.createElement("div"));

    const step = getStep("show-output");
    const promise = step.beforeShowPromise!();

    // fitView animates the viewport for 800ms; the step awaits a settle delay
    // before resolving so Shepherd measures the output at its final position.
    await vi.runAllTimersAsync();
    await promise;

    expect(waitForElement).toHaveBeenCalledWith(NODE_OUTPUT_SELECTOR, 20000);
    expect(fitViewToScreen).toHaveBeenCalledTimes(1);
  });

  it("resolves without throwing when the output never appears", async () => {
    vi.mocked(waitForElement).mockRejectedValue(new Error("not found"));

    const step = getStep("show-output");

    await expect(step.beforeShowPromise!()).resolves.toBeUndefined();
    expect(fitViewToScreen).not.toHaveBeenCalled();
  });

  it("highlights the node output on show and clears it on hide", () => {
    const step = getStep("show-output");

    step.when!.show!();
    expect(highlightElement).toHaveBeenCalledWith(NODE_OUTPUT_SELECTOR);

    step.when!.hide!();
    expect(removeAllHighlights).toHaveBeenCalled();
  });
});
