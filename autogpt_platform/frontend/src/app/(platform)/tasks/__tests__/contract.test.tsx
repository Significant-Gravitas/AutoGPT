import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, test, vi } from "vitest";
import { TaskPanel } from "../components/TaskPanel/TaskPanel";
import { TasksWorkspace } from "../components/TasksWorkspace/TasksWorkspace";

const mockPush = vi.hoisted(() => vi.fn());
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: mockPush }),
  usePathname: () => "/tasks",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

const minutesAgo = (n: number) =>
  new Date(Date.now() - n * 60_000).toISOString();

const OTTO = { type: "otto", id: null, name: "Otto" };

function row(id: string, number: number, extra: Record<string, unknown>) {
  return {
    id,
    number,
    title: `Task ${number}`,
    description: "",
    status: "in_progress",
    priority: "none",
    assignee: OTTO,
    labels: [],
    subtask_counts: { total: 0, done: 0 },
    metrics: { tool_calls: 0, tokens: 0, cost_usd: 0, duration_ms: 0 },
    live: null,
    pending_input: null,
    raw_status: "running",
    state: "working",
    now: "",
    ask: null,
    blocker: null,
    created_at: minutesAgo(30),
    updated_at: minutesAgo(1),
    ...extra,
  };
}

const PRICING = row("t1", 1, {
  title: "Check competitor pricing",
  now: "Pricing 4 of 6 tools",
});

const LAUNCH = row("t2", 2, {
  title: "Launch pricing page",
  status: "blocked",
  raw_status: "asked",
  state: "needs_you",
  now: "Waiting on your OK",
  updated_at: minutesAgo(3),
  ask: {
    id: "ask-1",
    question: "Publish the pricing page now?",
    why: "Both reviews passed.",
    options: ["Publish now", "Wait until Monday"],
    default_option: "Publish now",
  },
});

const BLOCKER = {
  done_so_far: "2 of 5 experts hired.",
  what_stopped_it:
    "Creating experts isn't permitted in a background run, so the last three couldn't be hired.",
  short: "Creating experts isn't permitted in a background run.",
  cause: "permission",
  next_actions: ["allow", "do_in_chat", "take_over"],
};

const HIRE = row("t3", 3, {
  title: "Hire 5 experts",
  status: "blocked",
  raw_status: "blocked",
  state: "stuck",
  now: BLOCKER.short,
  blocker: BLOCKER,
});

function serveBoard() {
  server.use(
    http.get("*/api/tasks/experts", () => HttpResponse.json({ experts: [] })),
    http.get("*/api/tasks", () =>
      HttpResponse.json({
        tasks: [PRICING, LAUNCH, HIRE],
        counts: {
          all: 3,
          running: 1,
          waiting: 2,
          done: 0,
          failed: 0,
          needs_you: 2,
          by_status: {},
          by_state: { working: 1, needs_you: 1, stuck: 1, done: 0, stopped: 0 },
        },
      }),
    ),
  );
}

function capture(method: "post", path: string) {
  const calls: { url: string; body: unknown }[] = [];
  server.use(
    http[method](path, async ({ request }) => {
      calls.push({ url: request.url, body: await request.json() });
      return HttpResponse.json({
        ...SUMMARY,
        work_session_id: "chat-9",
      });
    }),
  );
  return calls;
}

const SUMMARY = {
  id: "t3",
  key: "T-3",
  title: "Hire 5 experts",
  status: "blocked",
  state: "stuck",
  raw_status: "blocked",
  now: BLOCKER.short,
  work_session_id: null,
  version: 4,
  derived: false,
  live: { running: false, current_step: null },
  metrics: { tool_calls: 0, tokens: 0, cost_usd: 0, duration_ms: 0 },
  subtask_counts: { total: 0, done: 0 },
  updated_at: minutesAgo(1),
  contract: {
    goal: "Hire a designer, copywriter, analyst, developer and support lead",
    previous_goal: "Hire 4 experts",
    done_when: [
      {
        id: "c1",
        text: "Designer and copywriter hired",
        status: "met",
        evidence_ids: ["e1"],
      },
      { id: "c2", text: "All five hired", status: "pending" },
    ],
    plan: [
      {
        id: "s1",
        title: "Hire the team",
        status: "failed",
        rows: [
          { id: "r1", title: "Designer", status: "done" },
          { id: "r2", title: "Copywriter", status: "done" },
          { id: "r3", title: "Analyst", status: "failed" },
        ],
      },
    ],
    evidence: [
      {
        id: "e1",
        kind: "change",
        title: "Team update",
        summary: "Added Designer and Copywriter to your team",
        source_activity_id: "act-1",
      },
    ],
    result: null,
    ask: null,
    blocker: { ...BLOCKER, next_actions: ["retry_failed_part", "take_over"] },
  },
};

describe("task cards", () => {
  test("each card shows its live line with how long ago", async () => {
    serveBoard();
    render(<TasksWorkspace />);

    const pricing = (
      await screen.findByText("Check competitor pricing")
    ).closest("[data-testid='inbox-row']") as HTMLElement;
    expect(within(pricing).getByText("Pricing 4 of 6 tools")).toBeTruthy();
    expect(within(pricing).getByText("· 1m")).toBeTruthy();

    const tabs = screen.getByRole("radiogroup", { name: "Show tasks" });
    expect(within(tabs).getByText("Needs you")).toBeTruthy();
  });

  test("an open ask is answered from the card with its default", async () => {
    serveBoard();
    const calls = capture("post", "*/api/tasks/t2/ask/answer");
    const user = userEvent.setup();
    render(<TasksWorkspace />);

    const card = (await screen.findByText("Launch pricing page")).closest(
      "[data-testid='inbox-row']",
    ) as HTMLElement;
    expect(
      within(card).getByText("Publish the pricing page now?"),
    ).toBeTruthy();
    expect(within(card).getByText("Waiting on your OK")).toBeTruthy();

    await user.click(within(card).getByRole("button", { name: "Publish now" }));

    await waitFor(() =>
      expect(calls.map((c) => c.body)).toEqual([
        { answer: "Publish now", ask_id: "ask-1" },
      ]),
    );
  });

  test("a stuck card offers its first way out", async () => {
    serveBoard();
    const calls = capture("post", "*/api/tasks/t3/actions/:action");
    const user = userEvent.setup();
    render(<TasksWorkspace />);

    const card = (await screen.findByText("Hire 5 experts")).closest(
      "[data-testid='inbox-row']",
    ) as HTMLElement;
    expect(within(card).getByRole("img", { name: "Stuck" })).toBeTruthy();
    expect(within(card).getByText(BLOCKER.short)).toBeTruthy();

    await user.click(within(card).getByRole("button", { name: "Allow" }));

    await waitFor(() => expect(calls).toHaveLength(1));
    expect(calls[0].url).toMatch(/\/api\/tasks\/t3\/actions\/allow$/);
  });

  test("a stopped card offers no way out", async () => {
    server.use(
      http.get("*/api/tasks/experts", () => HttpResponse.json({ experts: [] })),
      http.get("*/api/tasks", () =>
        HttpResponse.json({
          tasks: [
            row("t5", 5, {
              title: "Summarise #support",
              status: "cancelled",
              raw_status: "cancelled",
              state: "stopped",
              now: "Stopped",
              blocker: BLOCKER,
            }),
          ],
        }),
      ),
    );
    render(<TasksWorkspace />);

    const card = (await screen.findByText("Summarise #support")).closest(
      "[data-testid='inbox-row']",
    ) as HTMLElement;
    expect(within(card).getByText("Stopped")).toBeTruthy();
    expect(within(card).queryByRole("button", { name: "Allow" })).toBeNull();
  });
});

describe("task panel", () => {
  function servePanel() {
    server.use(
      http.get("*/api/tasks/experts", () => HttpResponse.json({ experts: [] })),
      http.get("*/api/tasks/t3/summary", () => HttpResponse.json(SUMMARY)),
      http.get("*/api/tasks/t3/evidence/e1", () =>
        HttpResponse.json({
          evidence: SUMMARY.contract.evidence[0],
          activity: {
            id: "act-1",
            task_id: "t3",
            created_at: minutesAgo(5),
            actor: OTTO,
            kind: "evidence_added",
            body: "Added Designer and Copywriter",
            data: { content: "Designer: Ada\nCopywriter: Wren" },
          },
        }),
      ),
      http.get("*/api/tasks/t3", () =>
        HttpResponse.json({ ...HIRE, attachments: [], subtasks: [] }),
      ),
    );
  }

  test("shows the goal, what stopped it, the checks with proof and the parallel rows", async () => {
    servePanel();
    const calls = capture("post", "*/api/tasks/t3/actions/:action");
    const user = userEvent.setup();
    render(<TaskPanel taskId="t3" onOpenTask={vi.fn()} />);

    expect(await screen.findByText(SUMMARY.contract.goal)).toBeTruthy();
    expect(
      screen.getByText("Goal updated · was “Hire 4 experts”"),
    ).toBeTruthy();

    const stuck = screen.getByRole("region", { name: "What stopped it" });
    expect(within(stuck).getByText(BLOCKER.what_stopped_it)).toBeTruthy();
    expect(
      within(stuck).getByText("So far: 2 of 5 experts hired."),
    ).toBeTruthy();

    const checks = screen.getByRole("region", { name: "Done when" });
    expect(within(checks).getByText("1 of 2 met")).toBeTruthy();
    expect(within(checks).getByLabelText("Met")).toBeTruthy();
    await user.click(within(checks).getByRole("button", { name: "View" }));
    expect(await within(checks).findByText(/Designer: Ada/)).toBeTruthy();

    const progress = screen.getByRole("region", { name: "Progress" });
    expect(within(progress).getByText("2 of 3")).toBeTruthy();
    expect(within(progress).getByText("Analyst")).toBeTruthy();

    expect(screen.queryByRole("region", { name: "Sub-tasks" })).toBeNull();
    expect(screen.queryByRole("region", { name: "Description" })).toBeNull();

    await user.click(
      within(stuck).getByRole("button", { name: "Retry the failed part" }),
    );
    await waitFor(() => expect(calls).toHaveLength(1));
    expect(calls[0].url).toMatch(/\/actions\/retry_failed_part$/);
  });

  test("a stopped task no longer asks for anything", async () => {
    servePanel();
    server.use(
      http.get("*/api/tasks/t3/summary", () =>
        HttpResponse.json({
          ...SUMMARY,
          status: "cancelled",
          state: "stopped",
          raw_status: "cancelled",
        }),
      ),
    );
    render(<TaskPanel taskId="t3" onOpenTask={vi.fn()} />);

    expect(await screen.findByText(SUMMARY.contract.goal)).toBeTruthy();
    expect(
      screen.queryByRole("region", { name: "What stopped it" }),
    ).toBeNull();
    expect(screen.queryByRole("region", { name: "Needs you" })).toBeNull();
  });

  test("work waiting for your OK says what it delivered and what it missed", async () => {
    const result = "Email and LinkedIn post drafted, nothing sent.";
    const review = {
      ...SUMMARY,
      id: "t4",
      status: "in_review",
      state: "needs_you",
      raw_status: "awaiting_accept",
      contract: {
        ...SUMMARY.contract,
        previous_goal: null,
        done_when: [
          { id: "c1", text: "Post drafted", status: "met" },
          { id: "c2", text: "Email saved as a Gmail draft", status: "unmet" },
        ],
        plan: [],
        evidence: [],
        result: { summary: result },
        blocker: null,
      },
    };
    server.use(
      http.get("*/api/tasks/experts", () => HttpResponse.json({ experts: [] })),
      http.get("*/api/tasks/t4/summary", () => HttpResponse.json(review)),
      http.get("*/api/tasks/t4", () =>
        HttpResponse.json({
          ...row("t4", 4, {
            status: "in_review",
            raw_status: "awaiting_accept",
            state: "needs_you",
          }),
          attachments: [],
          subtasks: [],
        }),
      ),
    );
    render(<TaskPanel taskId="t4" onOpenTask={vi.fn()} />);

    const card = await screen.findByRole("region", { name: "Needs you" });
    expect(await within(card).findByText(result)).toBeTruthy();
    expect(
      within(card).getByText("Not met: Email saved as a Gmail draft"),
    ).toBeTruthy();
    expect(within(card).getByRole("button", { name: "Accept" })).toBeTruthy();
  });
});
