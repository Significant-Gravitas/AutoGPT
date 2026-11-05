import { getGetBrainDumpSuggestedTasksQueryKey } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import {
  type getBrainDumpSuggestedTasksResponse,
  useGetBrainDumpSuggestedTasks,
} from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import {
  getGetBrainDumpPlanMockHandler200,
  getGetBrainDumpRecommendedProvidersMockHandler200,
  getGetBrainDumpSuggestedTasksMockHandler200,
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

const { toast } = vi.hoisted(() => ({ toast: vi.fn() }));
vi.mock("@/components/molecules/Toast/use-toast", () => ({ toast }));

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

function makeWrapper(
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } }),
) {
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
  toast.mockClear();
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
const PLAN_WAIT_MS = 10_000;
const PLAN_POLL_MS = 1_000;
const HIRE_RETRY_MS = 2_000;

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

  it("never blocks Home on a hire that keeps failing, and says who did not join", async () => {
    const hires = mockHire(() =>
      HttpResponse.json({ detail: "boom" }, { status: 500 }),
    );
    const onComplete = vi.fn();
    renderPreparing({ onComplete, canHire: true });

    await advance(START_DELAY_MS + HIRE_MAX_WAIT_MS + 500);

    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(hires.count).toBeGreaterThan(1);
    expect(toast).toHaveBeenCalledTimes(1);
    expect(toast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Couldn't hire Maria, Max and Mina",
        description:
          "Otto takes on their tasks for now. You can hire experts any time.",
        variant: "destructive",
      }),
    );
  });

  it("tries the hire again when the server could not answer", async () => {
    let calls = 0;
    const hires = mockHire(() => {
      calls += 1;
      return calls === 1
        ? HttpResponse.json(
            { detail: "Subscription state temporarily unavailable" },
            { status: 503 },
          )
        : HttpResponse.json({
            hired: [
              { template_id: "tpl-maria", expert_id: "e-1", name: "Maria" },
            ],
            failed: [],
          });
    });
    renderPreparing({ canHire: true });

    await advance(START_DELAY_MS + HIRE_RETRY_MS + 200);

    expect(hires.count).toBe(2);
    expect(toast).not.toHaveBeenCalled();
  });

  it("does not try a refused hire again", async () => {
    const hires = mockHire(() =>
      HttpResponse.json({ detail: "Payment required" }, { status: 402 }),
    );
    renderPreparing({ canHire: true });

    await advance(START_DELAY_MS + HIRE_MAX_WAIT_MS);

    expect(hires.count).toBe(1);
  });

  it("says who could not join when the rest did", async () => {
    mockHire(() =>
      HttpResponse.json({
        hired: [{ template_id: "tpl-maria", expert_id: "e-1", name: "Maria" }],
        failed: [
          { template_id: "tpl-max", name: "Max" },
          { template_id: "tpl-mina", name: "Mina" },
        ],
      }),
    );
    renderPreparing({ canHire: true });

    await advance(START_DELAY_MS + 100);

    expect(toast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Couldn't hire Max and Mina",
        variant: "destructive",
      }),
    );
  });

  it("leaves reporting the hire to the backend", async () => {
    const capture = vi.spyOn(posthog, "capture");
    mockHire();
    renderPreparing({ canHire: true });

    await advance(START_DELAY_MS + 100);

    expect(capture).not.toHaveBeenCalledWith(
      "plan_experts_hired",
      expect.anything(),
    );
  });

  it("waits for the plan before hiring when the plan step showed none", async () => {
    let asked = 0;
    server.use(
      getGetBrainDumpPlanMockHandler200(() => {
        asked += 1;
        const plan = teamPlan(["Maria", "Max", "Mina"]);
        return asked < 3 ? { ...plan, ready: false } : plan;
      }),
    );
    const hires = mockHire();
    const { result } = renderPreparing({ canHire: true });

    await advance(START_DELAY_MS);
    expect(hires.count).toBe(0);
    expect(result.current.checklist[1]).toBe("Setting up your team");

    await advance(PLAN_POLL_MS * 2);
    expect(hires.count).toBe(1);
    expect(result.current.checklist[1]).toBe("Hiring Maria, Max and Mina");
  });

  it("hires anyway once the wait for the plan runs out", async () => {
    server.use(
      getGetBrainDumpPlanMockHandler200({
        ...teamPlan(["Maria", "Max", "Mina"]),
        ready: false,
      }),
    );
    const hires = mockHire();
    const onComplete = vi.fn();
    renderPreparing({ onComplete, canHire: true });

    await advance(PLAN_WAIT_MS - 500);
    expect(hires.count).toBe(0);

    await advance(1_000);
    expect(hires.count).toBe(1);

    await advance(GENERIC_TOTAL_MS);
    expect(onComplete).toHaveBeenCalledTimes(1);
  });

  it("gives Home's tasks to the new team when the hire lands after Home read them", async () => {
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false, staleTime: 60_000 } },
    });
    let expertID: string | null = null;
    server.use(
      getGetBrainDumpSuggestedTasksMockHandler200(() => ({
        stored: true,
        tasks: [
          {
            id: "0",
            title: "Draft the launch post",
            prompt: "Draft the launch post",
            expert_template_id: "tpl-maria",
            expert_name: "Maria",
            expert_id: expertID,
          },
        ],
      })),
    );
    let answer: (() => void) | undefined;
    mockHire(
      () =>
        new Promise<Response>((resolve) => {
          answer = () =>
            resolve(
              HttpResponse.json({
                hired: [
                  { template_id: "tpl-maria", expert_id: "e-1", name: "Maria" },
                ],
                failed: [],
              }),
            );
        }),
    );
    const preparing = renderHook(
      () => usePreparingStep({ onComplete: vi.fn(), canHire: true }),
      { wrapper: makeWrapper(client) },
    );
    await advance(START_DELAY_MS + HIRE_MAX_WAIT_MS + 500);
    preparing.unmount();

    const home = renderHook(() => useGetBrainDumpSuggestedTasks(), {
      wrapper: makeWrapper(client),
    });
    await advance(100);
    expect(firstOwner(home.result.current.data)).toBeNull();

    expertID = "e-1";
    await act(async () => answer?.());
    await advance(100);
    expect(firstOwner(home.result.current.data)).toBe("e-1");
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

  it("asks how the first hire went once its time is up, and refreshes Home", async () => {
    let calls = 0;
    const hires = mockHire(() => {
      calls += 1;
      return calls === 1
        ? new Promise<Response>(() => {})
        : HttpResponse.json({
            hired: [],
            failed: [{ template_id: "tpl-max", name: "Max" }],
          });
    });
    const first = renderPreparing({ canHire: true });
    await advance(START_DELAY_MS);
    first.unmount();

    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const invalidate = vi.spyOn(client, "invalidateQueries");
    renderHook(() => usePreparingStep({ onComplete: vi.fn(), canHire: true }), {
      wrapper: ({ children }: { children: ReactNode }) =>
        createElement(QueryClientProvider, { client }, children),
    });
    await advance(1_000);
    expect(hires.count).toBe(1);

    await advance(HIRE_MAX_WAIT_MS);

    expect(hires.count).toBe(2);
    expect(toast).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Couldn't hire Max" }),
    );
    expect(invalidate).toHaveBeenCalledWith({
      queryKey: getGetBrainDumpSuggestedTasksQueryKey(),
    });
  });

  it("says at once who could not join when nobody could, without trying again", async () => {
    const hires = mockHire(() =>
      HttpResponse.json({
        hired: [],
        failed: [{ template_id: "tpl-maria", name: "Maria" }],
      }),
    );
    const onComplete = vi.fn();
    renderPreparing({ onComplete, canHire: true });

    await advance(START_DELAY_MS);
    await advance(GENERIC_TOTAL_MS + 100);

    expect(hires.count).toBe(1);
    expect(toast).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Couldn't hire Maria" }),
    );
    expect(onComplete).toHaveBeenCalledTimes(1);
  });

  it("stays quiet when asking after a refresh fails, and still refreshes Home", async () => {
    let calls = 0;
    const hires = mockHire(() => {
      calls += 1;
      return calls === 1
        ? new Promise<Response>(() => {})
        : HttpResponse.json({ detail: "busy" }, { status: 503 });
    });
    const first = renderPreparing({ canHire: true });
    await advance(START_DELAY_MS);
    first.unmount();

    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const invalidate = vi.spyOn(client, "invalidateQueries");
    const onComplete = vi.fn();
    renderHook(() => usePreparingStep({ onComplete, canHire: true }), {
      wrapper: ({ children }: { children: ReactNode }) =>
        createElement(QueryClientProvider, { client }, children),
    });
    await advance(START_DELAY_MS);
    await advance(HIRE_MAX_WAIT_MS + 100);
    expect(onComplete).toHaveBeenCalledTimes(1);

    await advance(HIRE_MAX_WAIT_MS);

    expect(hires.count).toBeGreaterThan(2);
    expect(toast).not.toHaveBeenCalled();
    expect(invalidate).toHaveBeenCalledWith({
      queryKey: getGetBrainDumpSuggestedTasksQueryKey(),
    });
  });

  it("keeps trying when the re-ask after a refresh is the one that does the hire", async () => {
    let calls = 0;
    const hires = mockHire(() => {
      calls += 1;
      if (calls === 1) return new Promise<Response>(() => {});
      return calls === 2
        ? HttpResponse.json({ detail: "busy" }, { status: 503 })
        : HttpResponse.json({
            hired: [],
            failed: [{ template_id: "tpl-max", name: "Max" }],
          });
    });
    const first = renderPreparing({ canHire: true });
    await advance(START_DELAY_MS);
    first.unmount();

    renderPreparing({ canHire: true });
    await advance(HIRE_MAX_WAIT_MS);
    await advance(HIRE_RETRY_MS + 100);

    expect(hires.count).toBe(3);
    expect(toast).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Couldn't hire Max" }),
    );
  });

  it("says who did not join when the plan never settles inside the window", async () => {
    const hires = mockHire(() =>
      HttpResponse.json({ detail: "changing" }, { status: 409 }),
    );
    renderPreparing({ canHire: true });

    await advance(START_DELAY_MS + HIRE_MAX_WAIT_MS + 500);

    expect(hires.count).toBeGreaterThan(1);
    expect(toast).toHaveBeenCalledTimes(1);
    expect(toast).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Couldn't hire Maria, Max and Mina" }),
    );
  });

  it("tries again when the plan is still changing", async () => {
    let calls = 0;
    const hires = mockHire(() => {
      calls += 1;
      return calls === 1
        ? HttpResponse.json({ detail: "changing" }, { status: 409 })
        : HttpResponse.json({ hired: [], failed: [] });
    });
    renderPreparing({ canHire: true });

    await advance(START_DELAY_MS);
    await advance(HIRE_RETRY_MS + 100);

    expect(hires.count).toBe(2);
  });

  it("sends the hire again when the screen is refreshed between tries", async () => {
    let calls = 0;
    const hires = mockHire(() => {
      calls += 1;
      return calls === 1
        ? HttpResponse.json({ detail: "busy" }, { status: 503 })
        : new Promise<Response>(() => {});
    });
    const first = renderPreparing({ canHire: true });
    await advance(START_DELAY_MS);
    expect(hires.count).toBe(1);
    first.unmount();

    // Before the first screen's own retry would go out.
    renderPreparing({ canHire: true });
    await advance(HIRE_RETRY_MS - START_DELAY_MS - 500);

    expect(hires.count).toBe(2);
  });
});

function firstOwner(response: getBrainDumpSuggestedTasksResponse | undefined) {
  return response?.status === 200
    ? response.data.tasks?.[0]?.expert_id
    : undefined;
}
