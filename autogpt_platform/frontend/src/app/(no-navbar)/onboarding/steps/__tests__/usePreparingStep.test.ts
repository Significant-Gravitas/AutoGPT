import {
  getGetBrainDumpPlanMockHandler200,
  getGetBrainDumpRecommendedProvidersMockHandler200,
} from "@/app/api/__generated__/endpoints/brain-dump/brain-dump.msw";
import type { PlanHireResponse } from "@/app/api/__generated__/models/planHireResponse";
import type { PlanResponse } from "@/app/api/__generated__/models/planResponse";
import { server } from "@/mocks/mock-server";
import { http, HttpResponse } from "msw";
import posthog from "posthog-js";
import { setIntroPath } from "@/services/onboarding/brain-dump-handoff";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook } from "@testing-library/react";
import { createElement, type ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useOnboardingWizardStore } from "../../store";
import { usePreparingStep } from "../usePreparingStep";

const START_DELAY_MS = 300;
const GENERIC_TOTAL_MS = 4_000;
const DUMP_TOTAL_MS = 10_000;
const RECOMMENDATIONS_MAX_WAIT_MS = 15_000;

const GENERIC_CHECKLIST = [
  "Personalizing your experience",
  "Setting up your team",
  "Building your space",
];
const DUMP_CHECKLIST = [
  "Reading your brain dump",
  "Setting up your team",
  "Building your space",
  "Finding tools for your work",
];

function makeWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return function Wrapper({ children }: { children: ReactNode }) {
    return createElement(QueryClientProvider, { client }, children);
  };
}

function renderPreparing(args: { onComplete?: () => void; canHire?: boolean }) {
  return renderHook(
    () =>
      usePreparingStep({
        onComplete: args.onComplete ?? vi.fn(),
        canHire: args.canHire,
      }),
    { wrapper: makeWrapper() },
  );
}

function mockRecommendations(ready: boolean) {
  server.use(
    getGetBrainDumpRecommendedProvidersMockHandler200({
      ready,
      providers: [],
    }),
  );
}

async function advance(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

beforeEach(() => {
  window.sessionStorage.clear();
  useOnboardingWizardStore.setState({ userID: null });
  mockRecommendations(true);
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("usePreparingStep — checklist copy", () => {
  it("uses the honest dump copy only when the user actually dumped", () => {
    setIntroPath("A");

    const { result } = renderPreparing({});

    expect(result.current.checklist).toEqual(DUMP_CHECKLIST);
  });

  it("uses the generic copy on the skip path", () => {
    setIntroPath("B");
    const skipped = renderPreparing({});
    expect(skipped.result.current.checklist).toEqual(GENERIC_CHECKLIST);
  });

  it("does not consume the intro path — the copilot home still needs it", () => {
    setIntroPath("A");

    renderPreparing({});

    expect(window.sessionStorage.getItem("autogpt:onboarding-intro-path")).toBe(
      "A",
    );
  });
});

describe("usePreparingStep — progress", () => {
  it("holds everything still until the entrance delay elapses", async () => {
    const { result } = renderPreparing({});

    expect(result.current.started).toBe(false);
    expect(result.current.progress).toBe(0);
    expect(result.current.completedItems).toBe(0);

    await advance(START_DELAY_MS - 50);
    expect(result.current.started).toBe(false);
    expect(result.current.progress).toBe(0);

    await advance(50);
    expect(result.current.started).toBe(true);
  });

  it("fills the bar in proportion to elapsed time and ticks the checklist along", async () => {
    const { result } = renderPreparing({});

    await advance(START_DELAY_MS);
    await advance(GENERIC_TOTAL_MS / 2);

    expect(result.current.progress).toBe(50);
    expect(result.current.completedItems).toBe(1);

    await advance(GENERIC_TOTAL_MS / 4);

    expect(result.current.progress).toBe(75);
    expect(result.current.completedItems).toBe(2);
  });

  it("caps progress at 100 and the checklist at its own length", async () => {
    const { result } = renderPreparing({});

    await advance(START_DELAY_MS);
    await advance(GENERIC_TOTAL_MS + 5_000);

    expect(result.current.progress).toBe(100);
    expect(result.current.completedItems).toBe(result.current.checklist.length);
  });
});

describe("usePreparingStep — completion", () => {
  it("calls onComplete exactly once, and not before the full duration", async () => {
    const onComplete = vi.fn();
    renderPreparing({ onComplete });

    await advance(START_DELAY_MS);
    await advance(GENERIC_TOTAL_MS - 100);
    expect(onComplete).not.toHaveBeenCalled();

    await advance(100);
    expect(onComplete).toHaveBeenCalledTimes(1);

    await advance(5_000);
    expect(onComplete).toHaveBeenCalledTimes(1);
  });

  it("calls the latest onComplete, not the one captured on the first render", async () => {
    const stale = vi.fn();
    const fresh = vi.fn();
    const { rerender } = renderHook(
      ({ onComplete }) => usePreparingStep({ onComplete }),
      { initialProps: { onComplete: stale }, wrapper: makeWrapper() },
    );

    await advance(START_DELAY_MS + GENERIC_TOTAL_MS / 2);
    rerender({ onComplete: fresh });
    await advance(GENERIC_TOTAL_MS);

    expect(stale).not.toHaveBeenCalled();
    expect(fresh).toHaveBeenCalledTimes(1);
  });

  it("stops the ticker on unmount so a navigated-away wizard never completes", async () => {
    const onComplete = vi.fn();
    const { unmount } = renderPreparing({ onComplete });

    await advance(START_DELAY_MS + 1_000);
    unmount();
    await advance(GENERIC_TOTAL_MS);

    expect(onComplete).not.toHaveBeenCalled();
  });
});

describe("usePreparingStep — recommendation gate (dump path)", () => {
  it("completes after the full duration once recommendations are ready", async () => {
    setIntroPath("A");
    const onComplete = vi.fn();
    renderPreparing({ onComplete });

    await advance(START_DELAY_MS);
    await advance(DUMP_TOTAL_MS - 100);
    expect(onComplete).not.toHaveBeenCalled();

    await advance(200);
    expect(onComplete).toHaveBeenCalledTimes(1);
  });

  it("holds the last step and the bar while the recommendation job is still running", async () => {
    setIntroPath("A");
    mockRecommendations(false);
    const onComplete = vi.fn();
    const { result } = renderPreparing({
      onComplete,
    });

    await advance(START_DELAY_MS);
    // Short of the ceiling, so this asserts the gate rather than the
    // give-up timer that eventually overrides it.
    await advance(DUMP_TOTAL_MS + 2_000);

    expect(onComplete).not.toHaveBeenCalled();
    expect(result.current.completedItems).toBe(DUMP_CHECKLIST.length - 1);
    expect(result.current.progress).toBeLessThanOrEqual(95);
  });

  it("releases the gate on the poll that reports the job finished", async () => {
    setIntroPath("A");
    mockRecommendations(false);
    const onComplete = vi.fn();
    const { result } = renderPreparing({
      onComplete,
    });

    await advance(START_DELAY_MS);
    await advance(DUMP_TOTAL_MS + 500);
    expect(onComplete).not.toHaveBeenCalled();

    mockRecommendations(true);
    // Next poll (2.5s cadence) picks up ready=true; the response settles
    // at the end of the first window, so a second window lets the ticker
    // observe it and complete the run. Kept inside the give-up ceiling so
    // the poll is what releases the gate, not the timer.
    await advance(3_000);
    await advance(500);

    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(result.current.progress).toBe(100);
    expect(result.current.completedItems).toBe(DUMP_CHECKLIST.length);
  });

  it("never strands the user: advances anyway after the max wait ceiling", async () => {
    setIntroPath("A");
    mockRecommendations(false);
    const onComplete = vi.fn();
    renderPreparing({ onComplete });

    await advance(START_DELAY_MS);
    await advance(RECOMMENDATIONS_MAX_WAIT_MS + 1_000);

    expect(onComplete).toHaveBeenCalledTimes(1);
  });

  it("does not gate the generic path on the recommendation job", async () => {
    setIntroPath("B");
    mockRecommendations(false);
    const onComplete = vi.fn();
    renderPreparing({ onComplete });

    await advance(START_DELAY_MS);
    await advance(GENERIC_TOTAL_MS + 200);

    expect(onComplete).toHaveBeenCalledTimes(1);
  });
});

const HIRE_URL = "*/api/onboarding/brain-dump/plan/hire";
const HIRE_MAX_WAIT_MS = 15_000;

function teamPlan(names: string[]): PlanResponse {
  return {
    ready: true,
    source: "llm",
    skipped_dump: false,
    heard: "You sell stickers.",
    team: names.map((name) => ({
      template_id: `tpl-${name.toLowerCase()}`,
      name,
      role: "Marketing",
    })),
    tasks: [],
  };
}

// Counts every hire request; `respond` decides each answer.
function mockHire(
  respond: () => Promise<Response> | Response = () =>
    HttpResponse.json({
      hired: [{ template_id: "tpl-maria", expert_id: "e-1", name: "Maria" }],
      failed: [],
    } satisfies PlanHireResponse),
) {
  const calls = { count: 0 };
  server.use(
    http.post(HIRE_URL, () => {
      calls.count += 1;
      return respond();
    }),
  );
  return calls;
}

describe("usePreparingStep — hiring the plan's team", () => {
  beforeEach(() => {
    server.use(
      getGetBrainDumpPlanMockHandler200(teamPlan(["Maria", "Max", "Mina"])),
    );
  });

  it("hires once the trial is confirmed, naming the team in the checklist", async () => {
    const hires = mockHire();
    const { result } = renderPreparing({ canHire: true });

    await advance(START_DELAY_MS);

    expect(hires.count).toBe(1);
    expect(result.current.checklist[1]).toBe("Hiring Maria, Max and Mina");
  });

  it("never hires before the trial is confirmed", async () => {
    const hires = mockHire();
    const onComplete = vi.fn();
    const { result } = renderPreparing({ onComplete, canHire: false });

    await advance(START_DELAY_MS);
    await advance(GENERIC_TOTAL_MS + 100);

    expect(hires.count).toBe(0);
    expect(result.current.checklist[1]).toBe("Setting up your team");
    expect(onComplete).toHaveBeenCalledTimes(1);
  });

  it("says it is setting up the team when the plan has nobody to hire", async () => {
    server.use(getGetBrainDumpPlanMockHandler200(teamPlan([])));
    mockHire(() => HttpResponse.json({ hired: [], failed: [] }));
    const { result } = renderPreparing({ canHire: true });

    await advance(START_DELAY_MS);

    expect(result.current.checklist[1]).toBe("Setting up your team");
  });

  it("names the team the plan step showed, without asking again", async () => {
    useOnboardingWizardStore.setState({ userID: "u1" });
    window.sessionStorage.setItem(
      "autogpt:onboarding-plan:u1",
      JSON.stringify(teamPlan(["Nadia"])),
    );
    mockHire();
    const { result } = renderPreparing({ canHire: true });

    expect(result.current.checklist[1]).toBe("Hiring Nadia");
  });

  it("names the server's team when the plan step gave up waiting", async () => {
    useOnboardingWizardStore.setState({ userID: "u1" });
    window.sessionStorage.setItem(
      "autogpt:onboarding-plan:u1",
      JSON.stringify({ ...teamPlan(["Nadia"]), ready: false }),
    );
    mockHire();
    const { result } = renderPreparing({ canHire: true });

    await advance(START_DELAY_MS);

    expect(result.current.checklist[1]).toBe("Hiring Maria, Max and Mina");
  });

  it("holds the bar and the hire item until the hire answers", async () => {
    let answer: (() => void) | undefined;
    mockHire(
      () =>
        new Promise<Response>((resolve) => {
          answer = () => resolve(HttpResponse.json({ hired: [], failed: [] }));
        }),
    );
    const onComplete = vi.fn();
    const { result } = renderPreparing({ onComplete, canHire: true });

    await advance(START_DELAY_MS);
    await advance(GENERIC_TOTAL_MS + 2_000);

    expect(onComplete).not.toHaveBeenCalled();
    expect(result.current.completedItems).toBe(1);
    expect(result.current.progress).toBeLessThanOrEqual(95);

    await act(async () => answer?.());
    await advance(200);
    expect(onComplete).toHaveBeenCalledTimes(1);
  });

  it("moves on after the ceiling when the hire never answers", async () => {
    mockHire(() => new Promise<Response>(() => {}));
    const onComplete = vi.fn();
    renderPreparing({ onComplete, canHire: true });

    await advance(START_DELAY_MS);
    await advance(HIRE_MAX_WAIT_MS - 500);
    expect(onComplete).not.toHaveBeenCalled();

    await advance(1_000);
    expect(onComplete).toHaveBeenCalledTimes(1);
  });

  it("never blocks Home on a failed hire, and reports it", async () => {
    const capture = vi.spyOn(posthog, "capture");
    mockHire(() => HttpResponse.json({ detail: "boom" }, { status: 500 }));
    const onComplete = vi.fn();
    renderPreparing({ onComplete, canHire: true });

    await advance(START_DELAY_MS);
    await advance(GENERIC_TOTAL_MS + 200);

    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(capture).toHaveBeenCalledWith("plan_experts_hired", {
      hired: 0,
      failed: 3,
      via: "preparing",
    });
  });

  it("reports how many joined and how many could not", async () => {
    const capture = vi.spyOn(posthog, "capture");
    mockHire(() =>
      HttpResponse.json({
        hired: [{ template_id: "tpl-maria", expert_id: "e-1", name: "Maria" }],
        failed: [{ template_id: "tpl-max", name: "Max" }],
      }),
    );
    renderPreparing({ canHire: true });

    await advance(START_DELAY_MS + 100);

    expect(capture).toHaveBeenCalledWith("plan_experts_hired", {
      hired: 1,
      failed: 1,
      via: "preparing",
    });
  });

  it("does not send a second hire when the screen is refreshed mid-hire", async () => {
    const hires = mockHire(() => new Promise<Response>(() => {}));
    const first = renderPreparing({ canHire: true });
    await advance(START_DELAY_MS);
    first.unmount();

    renderPreparing({ canHire: true });
    await advance(START_DELAY_MS + 1_000);

    expect(hires.count).toBe(1);
  });
});
