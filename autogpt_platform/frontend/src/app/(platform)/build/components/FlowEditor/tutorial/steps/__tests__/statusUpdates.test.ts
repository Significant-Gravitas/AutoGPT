import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

const nodes = vi.hoisted(() => ({
  first: null as null | { id: string; data: { hardcodedValues: object } },
  second: null as null | { id: string; data: { hardcodedValues: object } },
}));

vi.mock(
  "@/app/(platform)/build/components/FlowEditor/tutorial/helpers",
  () => ({
    waitForElement: vi.fn(),
    waitForNodesCount: vi.fn(() => Promise.resolve()),
    fitViewToScreen: vi.fn(),
    highlightElement: vi.fn(),
    removeAllHighlights: vi.fn(),
    addSecondCalculatorBlock: vi.fn(),
    getFirstNode: () => nodes.first,
    getSecondCalculatorNode: () => nodes.second,
  }),
);

// Must import after mocks
import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";
import { createConfigureCalculatorSteps } from "../configure-calculator";
import { createSecondCalculatorSteps } from "../second-calculator";
import { createConnectionSteps } from "../connections";

type StepLike = {
  id?: string;
  text?: string;
  when?: { show?: () => void; hide?: () => void };
};

// The steps keep their edge subscription and mouseup handler on the current
// step, so it must be the same object on every call for hide to remove them.
const currentStep: Record<string, unknown> = {};

const tour = {
  next: vi.fn(),
  back: vi.fn(),
  show: vi.fn(),
  getCurrentStep: () => currentStep,
};

function mount(steps: StepLike[], id: string) {
  const step = steps.find((s) => s.id === id);
  if (!step) throw new Error(`step "${id}" not found`);
  document.body.innerHTML = step.text ?? "";
  return step;
}

function classes(selector: string) {
  return document.querySelector(selector)?.className.split(/\s+/) ?? [];
}

describe("tutorial requirement and connection status", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    for (const key of Object.keys(currentStep)) delete currentStep[key];
    nodes.first = null;
    nodes.second = null;
  });

  afterEach(() => {
    vi.useRealTimers();
    document.body.innerHTML = "";
    useEdgeStore.setState({ edges: [] });
  });

  it("turns the first calculator's requirements green, then back to amber", () => {
    const step = mount(
      createConfigureCalculatorSteps(tour) as StepLike[],
      "enter-values",
    );
    nodes.first = { id: "n1", data: { hardcodedValues: {} } };
    step.when?.show?.();

    vi.advanceTimersByTime(300);
    expect(classes("#req-a")).toContain("text-amber-600");
    expect(classes("#requirements-box")).toContain("bg-warning-surface");

    nodes.first = {
      id: "n1",
      data: { hardcodedValues: { a: 1, b: 2, operation: "Add" } },
    };
    vi.advanceTimersByTime(300);
    expect(classes("#req-a")).toContain("text-success");
    expect(classes("#requirements-box")).toEqual(
      expect.arrayContaining(["bg-success-surface", "ring-green-200"]),
    );
    expect(classes("#requirements-title")).toContain("text-success");

    nodes.first = { id: "n1", data: { hardcodedValues: { a: 1 } } };
    vi.advanceTimersByTime(300);
    expect(classes("#requirements-box")).toEqual(
      expect.arrayContaining(["bg-warning-surface", "ring-amber-200"]),
    );
    expect(classes("#requirements-title")).toContain("text-amber-600");
    step.when?.hide?.();
  });

  it("turns the second calculator's requirements green, then back to amber", () => {
    const step = mount(
      createSecondCalculatorSteps(tour) as StepLike[],
      "configure-second-calculator",
    );
    nodes.second = { id: "n2", data: { hardcodedValues: {} } };
    step.when?.show?.();

    vi.advanceTimersByTime(300);
    nodes.second = {
      id: "n2",
      data: { hardcodedValues: { b: 2, operation: "Add" } },
    };
    vi.advanceTimersByTime(300);
    expect(classes("#second-calc-requirements-box")).toEqual(
      expect.arrayContaining(["bg-success-surface", "ring-green-200"]),
    );
    expect(classes("#second-calc-requirements-title")).toContain(
      "text-success",
    );

    nodes.second = { id: "n2", data: { hardcodedValues: {} } };
    vi.advanceTimersByTime(300);
    expect(classes("#second-calc-requirements-box")).toEqual(
      expect.arrayContaining(["bg-warning-surface", "ring-amber-200"]),
    );
    step.when?.hide?.();
  });

  it("shows an existing connection as connected on both connection steps", () => {
    useEdgeStore.setState({
      edges: [{ id: "e1", source: "a", target: "b" }],
    } as never);
    const steps = createConnectionSteps(tour) as StepLike[];

    const output = mount(steps, "connect-blocks-output");
    output.when?.show?.();
    expect(classes("#connection-status-output")).toEqual(
      expect.arrayContaining(["bg-success-surface", "text-success"]),
    );
    output.when?.hide?.();

    const input = mount(steps, "connect-blocks-input");
    input.when?.show?.();
    expect(classes("#connection-status-input")).toEqual(
      expect.arrayContaining(["bg-success-surface", "text-success"]),
    );
    input.when?.hide?.();
  });

  it("turns both connection steps green when the edge is drawn after they show", () => {
    const steps = createConnectionSteps(tour) as StepLike[];

    for (const [id, status] of [
      ["connect-blocks-output", "#connection-status-output"],
      ["connect-blocks-input", "#connection-status-input"],
    ]) {
      useEdgeStore.setState({ edges: [] });
      tour.next.mockClear();
      const step = mount(steps, id);
      step.when?.show?.();
      expect(classes(status)).toEqual(
        expect.arrayContaining(["bg-warning-surface", "text-amber-600"]),
      );

      useEdgeStore.setState({
        edges: [{ id: "e1", source: "a", target: "b" }],
      } as never);
      expect(classes(status)).toEqual(
        expect.arrayContaining(["bg-success-surface", "text-success"]),
      );
      vi.advanceTimersByTime(500);
      expect(tour.next).toHaveBeenCalledTimes(1);
      step.when?.hide?.();
    }
  });
});
