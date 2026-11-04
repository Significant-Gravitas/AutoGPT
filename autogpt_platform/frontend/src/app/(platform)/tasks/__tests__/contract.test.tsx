import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, test, vi } from "vitest";
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
    expect(within(pricing).getByText("· 1m ago")).toBeTruthy();
    expect(within(pricing).getByTestId("task-status").textContent).toBe(
      "Working",
    );

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
    expect(within(card).getByTestId("task-status").textContent).toBe(
      "Needs you",
    );
    expect(within(card).queryByText("Waiting on your OK")).toBeNull();

    await user.click(within(card).getByRole("button", { name: "Publish now" }));

    await waitFor(() =>
      expect(calls.map((c) => c.body)).toEqual([
        { answer: "Publish now", ask_id: "ask-1" },
      ]),
    );
  });

  test("a failed card is red, says why and offers its first way out", async () => {
    serveBoard();
    const calls = capture("post", "*/api/tasks/t3/actions/:action");
    const user = userEvent.setup();
    render(<TasksWorkspace />);

    const card = (await screen.findByText("Hire 5 experts")).closest(
      "[data-testid='inbox-row']",
    ) as HTMLElement;
    expect(within(card).getByTestId("task-status").textContent).toBe("Failed");
    expect(within(card).getByText(BLOCKER.short)).toBeTruthy();

    await user.click(within(card).getByRole("button", { name: "Allow" }));

    await waitFor(() => expect(calls).toHaveLength(1));
    expect(calls[0].url).toMatch(/\/api\/tasks\/t3\/actions\/allow$/);
  });

  test("hire cards a task waits on are approved from its card", async () => {
    server.use(
      http.get("*/api/tasks/experts", () => HttpResponse.json({ experts: [] })),
      http.get("*/api/tasks", () =>
        HttpResponse.json({
          tasks: [
            row("t6", 6, {
              title: "Raise a developer and a marketer",
              status: "blocked",
              raw_status: "asked",
              state: "needs_you",
              now: "Waiting on your answer",
              ask: {
                id: "ask-6",
                question: "Click Approve on the cards, then reply done.",
                options: ["Done, approved both"],
                default_option: "Done, approved both",
              },
              approvals: [
                {
                  confirmation_id: "c-devin",
                  kind: "hire",
                  name: "Devin",
                  role: "Web Development",
                  tagline: "Builds and ships website changes.",
                },
                { confirmation_id: "c-maya", kind: "raise", name: "Maya" },
              ],
            }),
          ],
        }),
      ),
    );
    const calls = capture("post", "*/api/tasks/t6/approvals");
    const user = userEvent.setup();
    render(<TasksWorkspace />);

    const card = (
      await screen.findByText("Raise a developer and a marketer")
    ).closest("[data-testid='inbox-row']") as HTMLElement;
    expect(within(card).getByText("Hire Devin · Web Development")).toBeTruthy();
    expect(within(card).getByText("Create Maya")).toBeTruthy();
    // The cards replace the worker's own "reply done" question.
    expect(
      within(card).queryByRole("button", { name: "Done, approved both" }),
    ).toBeNull();

    await user.click(
      within(card).getByRole("button", { name: "Approve both" }),
    );

    await waitFor(() =>
      expect(calls.map((c) => c.body)).toEqual([
        {
          decisions: [
            { confirmation_id: "c-devin", approved: true },
            { confirmation_id: "c-maya", approved: true },
          ],
        },
      ]),
    );
  });

  test("a working task is cancelled from its card after one confirm", async () => {
    serveBoard();
    const calls = capture("post", "*/api/tasks/t1/actions/:action");
    const user = userEvent.setup();
    render(<TasksWorkspace />);

    const card = (await screen.findByText("Check competitor pricing")).closest(
      "[data-testid='inbox-row']",
    ) as HTMLElement;
    await user.click(within(card).getByRole("button", { name: "Cancel task" }));
    expect(calls).toHaveLength(0);

    await user.click(within(card).getByRole("button", { name: "Yes, cancel" }));

    await waitFor(() => expect(calls).toHaveLength(1));
    expect(calls[0].url).toMatch(/\/api\/tasks\/t1\/actions\/stop$/);
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
    expect(within(card).getByTestId("task-status").textContent).toBe(
      "Cancelled",
    );
    expect(within(card).queryByRole("button", { name: "Allow" })).toBeNull();
    expect(
      within(card).queryByRole("button", { name: "Cancel task" }),
    ).toBeNull();
  });
});

describe("task panel", () => {
  // An unserved route flips the client to its in-memory mocks for the rest
  // of the file, so every panel test serves an (empty) log.
  beforeEach(() => {
    server.use(
      http.get("*/api/tasks/:id/activity", () =>
        HttpResponse.json({ events: [], next_cursor: null }),
      ),
    );
  });

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

  test("a failed task says why in full, offers Retry, and keeps the bookkeeping under Show work", async () => {
    servePanel();
    const calls = capture("post", "*/api/tasks/t3/actions/:action");
    const user = userEvent.setup();
    render(<TaskPanel taskId="t3" onOpenTask={vi.fn()} />);

    const failed = await screen.findByRole("region", { name: "Failed" });
    expect(within(failed).getByText("Why:")).toBeTruthy();
    expect(within(failed).getByText(BLOCKER.what_stopped_it)).toBeTruthy();
    expect(
      within(failed).getByText("So far: 2 of 5 experts hired."),
    ).toBeTruthy();

    // The ask is empty, so nothing stands in for it; the goal, checks and
    // plan are bookkeeping and wait under Show work.
    expect(screen.queryByRole("region", { name: "What was asked" })).toBeNull();
    expect(screen.queryByText(SUMMARY.contract.goal)).toBeNull();
    expect(screen.queryByRole("region", { name: "Done when" })).toBeNull();
    expect(screen.queryByRole("region", { name: "Progress" })).toBeNull();
    expect(screen.queryByRole("region", { name: "Sub-tasks" })).toBeNull();

    await user.click(screen.getByRole("button", { name: "Show work" }));
    expect(await screen.findByText(SUMMARY.contract.goal)).toBeTruthy();
    expect(
      screen.getByText("Goal updated · was “Hire 4 experts”"),
    ).toBeTruthy();
    const checks = screen.getByRole("region", { name: "Done when" });
    expect(within(checks).getByText("1 of 2 met")).toBeTruthy();
    expect(within(checks).getByLabelText("Met")).toBeTruthy();
    await user.click(within(checks).getByRole("button", { name: "View" }));
    expect(await within(checks).findByText(/Designer: Ada/)).toBeTruthy();
    const progress = screen.getByRole("region", { name: "Progress" });
    expect(within(progress).getByText("2 of 3")).toBeTruthy();
    expect(within(progress).getByText("Analyst")).toBeTruthy();

    await user.click(within(failed).getByRole("button", { name: "Retry" }));
    await waitFor(() => expect(calls).toHaveLength(1));
    expect(calls[0].url).toMatch(/\/actions\/retry_failed_part$/);
  });

  test("a failure that needs a file offers Retry and Upload file", async () => {
    const reason =
      "Leo's turn failed: the billing export wasn't found. Upload the CSV from Stripe and retry, and Leo will pick it up from there without starting over.";
    const blocker = {
      done_so_far: "",
      what_stopped_it: reason,
      short: "The billing export wasn't found.",
      next_actions: ["Retry", "Upload file"],
    };
    server.use(
      http.get("*/api/tasks/experts", () => HttpResponse.json({ experts: [] })),
      http.get("*/api/tasks/t8/summary", () =>
        HttpResponse.json({
          ...SUMMARY,
          id: "t8",
          work_session_id: "chat-8",
          contract: {
            ...SUMMARY.contract,
            done_when: [],
            plan: [],
            evidence: [],
            blocker,
          },
        }),
      ),
      http.get("*/api/tasks/t8", () =>
        HttpResponse.json({
          ...row("t8", 8, {
            title: "Summarize the billing export",
            description: "Summarize the attached billing export",
            status: "blocked",
            raw_status: "blocked",
            state: "stuck",
            blocker,
          }),
          attachments: [],
          subtasks: [],
        }),
      ),
    );
    render(<TaskPanel taskId="t8" onOpenTask={vi.fn()} />);

    const asked = await screen.findByRole("region", { name: "What was asked" });
    expect(
      within(asked).getByText("Summarize the attached billing export"),
    ).toBeTruthy();
    const failed = screen.getByRole("region", { name: "Failed" });
    expect(within(failed).getByText(reason)).toBeTruthy();
    expect(within(failed).getByRole("button", { name: "Retry" })).toBeTruthy();
    expect(
      within(failed)
        .getByRole("link", { name: "Upload file" })
        .getAttribute("href"),
    ).toBe("/copilot?sessionId=chat-8");
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

    expect(
      await screen.findByRole("heading", { name: "Hire 5 experts" }),
    ).toBeTruthy();
    expect(screen.queryByRole("region", { name: "Failed" })).toBeNull();
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

  test("a done task shows its outcome once, in markdown, over a short feed", async () => {
    const MAYA = { type: "expert", id: "maya", name: "Maya" };
    const USER = { type: "user", id: "u1", name: "Abhi" };
    const result = "Picked **Linear** for the design team.";
    const done = {
      ...SUMMARY,
      id: "t7",
      status: "done",
      state: "done",
      raw_status: "done",
      metrics: { tool_calls: 2, tokens: 9000, cost_usd: 0.03, duration_ms: 0 },
      contract: {
        ...SUMMARY.contract,
        previous_goal: null,
        done_when: [{ id: "c1", text: "A tool picked", status: "met" }],
        plan: [],
        evidence: [
          {
            id: "e9",
            kind: "link",
            title: "Comparison",
            summary: "Compared **3** tools",
          },
        ],
        result: { summary: result, primary_evidence_id: "e9" },
        blocker: null,
      },
    };
    const at = (n: number) => minutesAgo(10 - n);
    const turn = (id: string, n: number, summary: string | null) => ({
      id,
      kind: "turn",
      created_at: at(n),
      actor: MAYA,
      data: {
        summary,
        tool_calls: 1,
        cost_usd: 0.01,
        tokens: 3000,
        tool_call_cards: [{ id: `${id}-c`, title: "Read file", status: "ok" }],
      },
    });
    server.use(
      http.get("*/api/tasks/experts", () => HttpResponse.json({ experts: [] })),
      http.get("*/api/tasks/t7/summary", () => HttpResponse.json(done)),
      http.get("*/api/tasks/t7/activity", () =>
        HttpResponse.json({
          events: [
            {
              id: "a1",
              kind: "created",
              created_at: at(0),
              actor: USER,
              body: "",
            },
            {
              id: "a2",
              kind: "status_changed",
              created_at: at(1),
              actor: MAYA,
              data: { from: "todo", to: "in_progress" },
            },
            {
              id: "a3",
              kind: "contract_set",
              created_at: at(2),
              actor: MAYA,
              body: "Pick a tool",
            },
            {
              id: "a4",
              kind: "contract_set",
              created_at: at(3),
              actor: MAYA,
              body: "Pick a tool",
            },
            turn("a5", 4, null),
            turn("a6", 5, "Drafted a **shortlist** of three tools for T-12"),
            {
              id: "a7",
              kind: "step_updated",
              created_at: at(6),
              actor: MAYA,
              body: "Compare",
              data: { status: "done" },
            },
            turn("a8", 7, result),
            {
              id: "a9",
              kind: "outcome",
              created_at: at(8),
              actor: MAYA,
              body: result,
            },
          ],
          next_cursor: null,
        }),
      ),
      http.get("*/api/tasks/t7", () =>
        HttpResponse.json({
          ...row("t7", 7, {
            title: "Pick a project tool",
            description: "Pick a project tool for the design team",
            status: "done",
            raw_status: "done",
            state: "done",
            outcome_summary: result,
            metrics: done.metrics,
          }),
          attachments: [],
          subtasks: [],
        }),
      ),
    );
    const user = userEvent.setup();
    render(<TaskPanel taskId="t7" onOpenTask={vi.fn()} />);

    const outcome = await screen.findByRole("region", { name: "Outcome" });
    expect(await within(outcome).findByText("Linear")).toBeTruthy();
    expect(within(outcome).getByText("3")).toBeTruthy();
    expect(outcome.textContent).not.toContain("**");
    expect(screen.getAllByRole("button", { name: "View" })).toHaveLength(1);

    const feed = await screen.findByRole("list", {
      name: "Activity on Pick a project tool",
    });
    await waitFor(() =>
      expect(within(feed).getAllByRole("listitem")).toHaveLength(3),
    );
    const lines = within(feed)
      .getAllByRole("listitem")
      .map((li) => li.textContent ?? "");
    expect(lines[0]).toMatch(/^You created the task/);
    expect(lines[1]).toMatch(/^Maya drafted a shortlist of three tools for/);
    expect(lines[2]).toMatch(/^Maya finished the task/);
    expect(lines.join(" ")).not.toMatch(/\*\*|Moved to|Read file|T-12|\$/);
    expect(screen.queryByText(/9k tokens/)).toBeNull();

    await user.click(screen.getByRole("button", { name: "Show work" }));
    expect(await screen.findByText(/9k tokens · \$0\.03/)).toBeTruthy();
    const steps = await screen.findByRole("list", {
      name: "Every step on Pick a project tool",
    });
    expect(within(steps).getAllByRole("listitem").length).toBeGreaterThan(5);
    expect(screen.getAllByRole("button", { name: "View" })).toHaveLength(1);
  });
});
