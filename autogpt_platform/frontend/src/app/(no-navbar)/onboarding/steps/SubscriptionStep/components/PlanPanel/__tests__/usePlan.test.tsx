import { getGetBrainDumpPlanQueryKey } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import {
  getGetBrainDumpPlanMockHandler200,
  getGetBrainDumpPlanMockHandler401,
  getSettleBrainDumpPlanMockHandler200,
  getSettleBrainDumpPlanMockHandler401,
} from "@/app/api/__generated__/endpoints/brain-dump/brain-dump.msw";
import type { PlanResponse } from "@/app/api/__generated__/models/planResponse";
import { server } from "@/mocks/mock-server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { useOnboardingWizardStore } from "../../../../../store";
import { usePlan } from "../usePlan";

const { posthog, searchParams } = vi.hoisted(() => ({
  posthog: { capture: vi.fn() },
  searchParams: { current: new URLSearchParams() },
}));

vi.mock("posthog-js", () => ({ default: posthog }));

// A skip from the brain dump step that has not reached the server yet.
const skip = vi.hoisted(() => ({ current: Promise.resolve() }));
vi.mock("../../../../BrainDumpStep/useBrainDumpStep", () => ({
  whenSkipSettled: () => skip.current,
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/onboarding",
  useSearchParams: () => searchParams.current,
  useParams: () => ({}),
}));

const POLL_MS = 1_000;
const MAX_WAIT_MS = 30_000;
// Long enough for the first answer to land, short of the first poll after it.
const FIRST_ANSWER_MS = 100;
const USER_ID = "user-1";
const SHOWN_PLAN_KEY = `autogpt:onboarding-plan:${USER_ID}`;

function expert(name: string) {
  return {
    template_id: `tpl-${name.toLowerCase()}`,
    name,
    role: "Sales",
    avatar_url: null,
  };
}

function tasks(count: number, prefix: string) {
  return Array.from({ length: count }, (_, index) => ({
    id: `${prefix}-${index + 1}`,
    title: `${prefix} task ${index + 1}`,
    prompt: `Do ${prefix} task ${index + 1}`,
    expert_template_id: null,
  }));
}

// The role plan the endpoint carries while the pipeline is still writing.
const WRITING: PlanResponse = {
  ready: false,
  source: "fallback",
  skipped_dump: true,
  heard: null,
  team: [expert("Maria")],
  tasks: tasks(2, "role"),
};

const READY: PlanResponse = {
  ready: true,
  source: "llm",
  skipped_dump: false,
  heard: "You run a bakery and lose your evenings to invoices.",
  team: [expert("Maria"), expert("Max")],
  tasks: tasks(5, "llm"),
};

const OTHER_READY: PlanResponse = {
  ...READY,
  heard: "You run a bike shop and want more repeat customers.",
  tasks: tasks(3, "other"),
};

// The same role plan once the server has stored it as the take's plan.
const SETTLED: PlanResponse = { ...WRITING, ready: true };

const EMPTY_PLAN: PlanResponse = {
  ready: false,
  source: "fallback",
  skipped_dump: false,
  heard: null,
  team: [],
  tasks: [],
};

function makeClient() {
  return new QueryClient({ defaultOptions: { queries: { retry: false } } });
}

// Answers in order, repeating the last one; counts what was asked.
function servePlans(...answers: PlanResponse[]) {
  const requests = { count: 0 };
  server.use(
    getGetBrainDumpPlanMockHandler200(() => {
      const answer = answers[Math.min(requests.count, answers.length - 1)];
      requests.count += 1;
      return answer;
    }),
  );
  return requests;
}

function serveSettle(answer: PlanResponse | (() => Promise<PlanResponse>)) {
  const requests = { count: 0 };
  server.use(
    getSettleBrainDumpPlanMockHandler200(async () => {
      requests.count += 1;
      return typeof answer === "function" ? answer() : answer;
    }),
  );
  return requests;
}

function renderPlan(client = makeClient()) {
  const seen: (PlanResponse | null)[] = [];
  const view = renderHook(
    () => {
      const result = usePlan();
      seen.push(result.plan);
      return result;
    },
    {
      wrapper: function Wrapper({ children }: { children: ReactNode }) {
        return (
          <QueryClientProvider client={client}>{children}</QueryClientProvider>
        );
      },
    },
  );
  return { ...view, seen };
}

function planViewedCalls() {
  return posthog.capture.mock.calls.filter(
    ([event]) => event === "plan_viewed",
  );
}

async function advance(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

beforeEach(() => {
  skip.current = Promise.resolve();
  vi.useFakeTimers();
  posthog.capture.mockClear();
  searchParams.current = new URLSearchParams();
  sessionStorage.clear();
  useOnboardingWizardStore.getState().reset();
  useOnboardingWizardStore.setState({ userID: USER_ID });
});

afterEach(() => {
  vi.clearAllTimers();
  vi.useRealTimers();
  sessionStorage.clear();
});

describe("usePlan polling", () => {
  test("polls every second while the plan is being written", async () => {
    const requests = servePlans(WRITING);
    const { result } = renderPlan();

    await advance(FIRST_ANSWER_MS);
    expect(requests.count).toBe(1);
    expect(result.current.isPending).toBe(true);
    expect(result.current.plan).toBeNull();

    await advance(POLL_MS);
    expect(requests.count).toBe(2);

    await advance(POLL_MS);
    expect(requests.count).toBe(3);
    expect(result.current.plan).toBeNull();
  });

  test("shows the plan once it is ready and stops polling", async () => {
    const requests = servePlans(WRITING, WRITING, READY);
    const { result } = renderPlan();

    await advance(FIRST_ANSWER_MS + POLL_MS);
    expect(result.current.plan).toBeNull();

    await advance(POLL_MS);
    expect(requests.count).toBe(3);
    expect(result.current.isPending).toBe(false);
    expect(result.current.plan).toEqual(READY);

    await advance(POLL_MS * 4);
    expect(requests.count).toBe(3);
    expect(result.current.plan).toEqual(READY);
  });

  test("passes on how far the jobs have got until the plan is shown", async () => {
    const progress = { tasks_written: false, team_chosen: true };
    servePlans({ ...WRITING, progress }, READY);
    const { result } = renderPlan();

    await advance(FIRST_ANSWER_MS);
    expect(result.current.progress).toEqual(progress);

    await advance(POLL_MS);
    expect(result.current.plan).toEqual(READY);
    expect(result.current.progress).toBeNull();
  });
});

describe("usePlan giving up", () => {
  test("settles the plan at 30 s and shows the plan the server settled on", async () => {
    servePlans(WRITING);
    const settles = serveSettle(SETTLED);
    const { result } = renderPlan();

    await advance(MAX_WAIT_MS - 1_000);
    expect(result.current.isPending).toBe(true);
    expect(result.current.plan).toBeNull();
    expect(settles.count).toBe(0);

    await advance(1_000 + FIRST_ANSWER_MS);
    expect(settles.count).toBe(1);
    expect(result.current.isPending).toBe(false);
    expect(result.current.plan).toEqual(SETTLED);
  });

  test("shows the last answer when the settle fails", async () => {
    servePlans(WRITING);
    server.use(getSettleBrainDumpPlanMockHandler401());
    const { result } = renderPlan();

    await advance(MAX_WAIT_MS + FIRST_ANSWER_MS);

    expect(result.current.isPending).toBe(false);
    expect(result.current.plan).toEqual(WRITING);
  });

  test("gives up with an empty plan when no answer ever came", async () => {
    server.use(
      getGetBrainDumpPlanMockHandler401(),
      getSettleBrainDumpPlanMockHandler401(),
    );
    const { result } = renderPlan();

    await advance(MAX_WAIT_MS + FIRST_ANSWER_MS);

    expect(result.current.isPending).toBe(false);
    expect(result.current.plan).toEqual(EMPTY_PLAN);
  });

  test("stops polling after giving up and keeps the plan it showed", async () => {
    let answer = WRITING;
    const requests = { count: 0 };
    server.use(
      getGetBrainDumpPlanMockHandler200(() => {
        requests.count += 1;
        return answer;
      }),
    );
    serveSettle(SETTLED);
    const { result, seen } = renderPlan();

    await advance(MAX_WAIT_MS + 500);
    expect(result.current.plan).toEqual(SETTLED);
    const asked = requests.count;
    const shownFrom = seen.length;

    answer = READY;
    await advance(POLL_MS * 4);

    expect(requests.count).toBe(asked);
    expect(result.current.plan).toEqual(SETTLED);
    expect(seen.slice(shownFrom)).not.toContainEqual(READY);
  });

  test("keeps the settled plan when a poll in flight at 30 s lands later", async () => {
    let release!: () => void;
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    const requests = { count: 0 };
    server.use(
      getGetBrainDumpPlanMockHandler200(async () => {
        requests.count += 1;
        // The poll sent at 29 s answers only after the 30 s give-up.
        if (requests.count === 30) {
          await held;
          return READY;
        }
        return WRITING;
      }),
    );
    serveSettle(SETTLED);
    const { result } = renderPlan();

    await advance(MAX_WAIT_MS + 500);
    expect(requests.count).toBe(30);
    expect(result.current.plan).toEqual(SETTLED);

    release();
    await advance(POLL_MS);

    expect(result.current.plan).toEqual(SETTLED);
  });

  test("settles only once a skip still on its way has landed", async () => {
    servePlans(WRITING);
    const settles = serveSettle(SETTLED);
    const { result } = renderPlan();
    await advance(FIRST_ANSWER_MS);
    // Back, then Skip, with the wait about to run out.
    let land!: () => void;
    skip.current = new Promise<void>((resolve) => {
      land = resolve;
    });

    await advance(MAX_WAIT_MS);
    expect(settles.count).toBe(0);
    expect(result.current.plan).toBeNull();

    land();
    await advance(FIRST_ANSWER_MS);
    expect(settles.count).toBe(1);
    expect(result.current.plan).toEqual(SETTLED);
  });

  test("keeps a plan that lands while the settle is on its way", async () => {
    let release!: () => void;
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    servePlans(...Array<PlanResponse>(30).fill(WRITING), READY);
    serveSettle(async () => {
      await held;
      return SETTLED;
    });
    const { result } = renderPlan();

    await advance(MAX_WAIT_MS + 500);
    expect(result.current.plan).toEqual(READY);

    release();
    await advance(POLL_MS);

    expect(result.current.plan).toEqual(READY);
  });
});

describe("usePlan after a skip", () => {
  test("reads the plan only once a skip still on its way has landed", async () => {
    // Back from the plan step, then Skip: until the skip lands, the server
    // still answers for the take the user went Back from.
    let land!: () => void;
    skip.current = new Promise<void>((resolve) => {
      land = resolve;
    });
    const requests = servePlans(READY);
    const { result } = renderPlan();

    await advance(1_000);
    expect(requests.count).toBe(0);
    expect(result.current.plan).toBeNull();

    land();
    await advance(100);
    expect(requests.count).toBe(1);
    expect(result.current.plan).toEqual(READY);
  });
});

describe("usePlan plan_viewed", () => {
  test("reports the shown plan exactly once per mount", async () => {
    servePlans(WRITING, READY);
    const first = renderPlan();

    await advance(FIRST_ANSWER_MS + POLL_MS);
    expect(first.result.current.plan).toEqual(READY);
    expect(planViewedCalls()).toHaveLength(1);
    const [, properties] = planViewedCalls()[0];
    expect(properties).toEqual({
      plan_source: "llm",
      skipped_dump: false,
      tasks: 3,
      experts: 2,
      wait_ms: expect.any(Number),
      gave_up: false,
    });
    expect(properties.wait_ms).toBeGreaterThanOrEqual(POLL_MS);
    expect(properties.wait_ms).toBeLessThan(POLL_MS * 2);

    await advance(POLL_MS * 4);
    first.rerender();
    expect(planViewedCalls()).toHaveLength(1);

    first.unmount();
    servePlans(READY);
    const second = renderPlan();
    await advance(1_000);
    expect(second.result.current.plan).toEqual(READY);
    expect(planViewedCalls()).toHaveLength(2);
  });

  test("reports a given-up plan as given up", async () => {
    servePlans(WRITING);
    serveSettle(SETTLED);
    renderPlan();

    await advance(MAX_WAIT_MS + FIRST_ANSWER_MS);

    expect(planViewedCalls()).toHaveLength(1);
    const [, properties] = planViewedCalls()[0];
    expect(properties).toEqual({
      plan_source: "fallback",
      skipped_dump: true,
      tasks: 2,
      experts: 1,
      wait_ms: expect.any(Number),
      gave_up: true,
    });
    expect(properties.wait_ms).toBeGreaterThanOrEqual(MAX_WAIT_MS);
  });

  test("reports a plan that landed after the wait ran out as given up", async () => {
    let release!: () => void;
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    servePlans(...Array<PlanResponse>(30).fill(WRITING), READY);
    serveSettle(async () => {
      await held;
      return SETTLED;
    });
    renderPlan();

    await advance(MAX_WAIT_MS + 500);

    expect(planViewedCalls()).toHaveLength(1);
    expect(planViewedCalls()[0][1]).toMatchObject({
      plan_source: "llm",
      gave_up: true,
    });
    release();
    await advance(FIRST_ANSWER_MS);
  });

  test("reports nothing while the plan is still pending", async () => {
    servePlans(WRITING);
    renderPlan();

    await advance(MAX_WAIT_MS - 1_000);

    expect(planViewedCalls()).toHaveLength(0);
  });
});

describe("usePlan cached and remembered plans", () => {
  test("does not show a plan cached before the mount while the refetch is pending", async () => {
    const client = makeClient();
    client.setQueryData(getGetBrainDumpPlanQueryKey(), {
      status: 200,
      data: OTHER_READY,
      headers: new Headers(),
    });
    servePlans(WRITING, READY);
    const { result, seen } = renderPlan(client);

    expect(seen[0]).toBeNull();
    expect(result.current.isPending).toBe(true);

    await advance(FIRST_ANSWER_MS);
    expect(result.current.plan).toBeNull();

    await advance(POLL_MS);
    expect(result.current.plan).toEqual(READY);
    expect(seen).not.toContainEqual(OTHER_READY);
  });

  test("remembers the shown plan for this user", async () => {
    servePlans(WRITING, READY);
    renderPlan();

    await advance(FIRST_ANSWER_MS);
    expect(sessionStorage.getItem(SHOWN_PLAN_KEY)).toBeNull();

    await advance(POLL_MS);
    expect(JSON.parse(sessionStorage.getItem(SHOWN_PLAN_KEY)!)).toEqual(READY);
  });

  test.each(["subscription=cancelled", "trial=cancelled"])(
    "shows the remembered plan at once on a checkout return (%s)",
    async (query) => {
      searchParams.current = new URLSearchParams(query);
      sessionStorage.setItem(SHOWN_PLAN_KEY, JSON.stringify(READY));
      servePlans(READY);
      const { result, seen } = renderPlan();

      expect(seen[0]).toEqual(READY);
      expect(result.current.isPending).toBe(false);

      await advance(POLL_MS * 2);
      expect(result.current.plan).toEqual(READY);
      expect(seen).not.toContain(null);
    },
  );

  test("does not report a restored plan as a second view", async () => {
    searchParams.current = new URLSearchParams("trial=cancelled");
    sessionStorage.setItem(SHOWN_PLAN_KEY, JSON.stringify(READY));
    servePlans(READY);
    renderPlan();

    await advance(1_000);
    expect(planViewedCalls()).toHaveLength(0);
  });

  test("waits for the server when nothing was remembered", async () => {
    searchParams.current = new URLSearchParams("subscription=cancelled");
    servePlans(READY);
    const { result, seen } = renderPlan();

    expect(seen[0]).toBeNull();
    await advance(1_000);
    expect(result.current.plan).toEqual(READY);
  });

  test("ignores a remembered plan outside a checkout return", () => {
    sessionStorage.setItem(SHOWN_PLAN_KEY, JSON.stringify(READY));
    servePlans(WRITING);
    const { seen } = renderPlan();

    expect(seen[0]).toBeNull();
  });

  test("ignores another user's remembered plan", () => {
    searchParams.current = new URLSearchParams("trial=cancelled");
    sessionStorage.setItem(
      "autogpt:onboarding-plan:someone-else",
      JSON.stringify(READY),
    );
    servePlans(WRITING);
    const { seen } = renderPlan();

    expect(seen[0]).toBeNull();
  });

  test("shows a remembered given-up plan at once on a checkout return", async () => {
    searchParams.current = new URLSearchParams("subscription=cancelled");
    sessionStorage.setItem(SHOWN_PLAN_KEY, JSON.stringify(WRITING));
    servePlans(WRITING);
    const { result, seen } = renderPlan();

    expect(seen[0]).toEqual(WRITING);
    await advance(1_000);
    expect(result.current.plan).toEqual(WRITING);
  });

  test("keeps the remembered plan when the return's refetch answers another", async () => {
    searchParams.current = new URLSearchParams("subscription=cancelled");
    sessionStorage.setItem(SHOWN_PLAN_KEY, JSON.stringify(READY));
    servePlans(OTHER_READY);
    const { result, seen } = renderPlan();

    expect(seen[0]).toEqual(READY);
    await advance(1_000);
    expect(result.current.plan).toEqual(READY);
  });
});
