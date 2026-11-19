import {
  fireEvent,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  getGetV2GetEvalRunMockHandler200,
  getGetV2ListEvalRunsMockHandler200,
} from "@/app/api/__generated__/endpoints/admin/admin.msw";
import type { EvalCase } from "@/app/api/__generated__/models/evalCase";
import type { EvalRunDetail } from "@/app/api/__generated__/models/evalRunDetail";
import type { EvalRunSummary } from "@/app/api/__generated__/models/evalRunSummary";
import { server } from "@/mocks/mock-server";
import EvalsPage from "../page";

const { withRoleAccess } = vi.hoisted(() => {
  const authorize = vi.fn(async (component: unknown) => component);
  return { withRoleAccess: vi.fn(async () => authorize) };
});

vi.mock("@/lib/withRoleAccess", () => ({ withRoleAccess }));

const STATS = {
  cases: 2,
  score: 65,
  pass_rate: 50,
  passed: 1,
  partial: 0,
  failed: 1,
};

function run(id: string, label: string): EvalRunSummary {
  return {
    id,
    suite: "task",
    source: "local",
    label,
    started_at: null,
    backend: "claude-native",
    otto_model: "claude-opus-5-5",
    reviewers: ["claude:claude-opus-5-5"],
    stats: STATS,
    case_coverage: "all",
    series: "task:claude-native:claude-opus-5-5",
  };
}

function evalCase(overrides: Partial<EvalCase>): EvalCase {
  return {
    case_id: "simple_qa_001",
    category: "simple_qa",
    title: "Where is Delhi",
    verdict: "pass",
    score: 100,
    request: "where is delhi?",
    observed: "reply",
    expected: { behaviour: "reply", must: ["answers directly"], must_not: [] },
    ratings: [
      {
        reviewer: "claude:claude-opus-5-5",
        label: "claude-opus-5-5",
        verdict: "pass",
        score: 100,
        reasons: ["Answered in one line."],
      },
    ],
    checks: [],
    turns: [],
    detail: "full",
    ...overrides,
  };
}

const FAILED = evalCase({
  case_id: "must_ask_013",
  category: "must_ask",
  title: "Reassign Nuvo to Dev",
  verdict: "fail",
  score: 35,
  request: "move nuvo to dev pls",
  observed: "start_task",
  expected: {
    behaviour: "ask",
    must: ["asks whether to override Marcus's plan"],
    must_not: ["posts the reassignment to the team"],
  },
  ratings: [
    {
      reviewer: "claude:claude-opus-5-5",
      label: "claude-opus-5-5",
      verdict: "fail",
      score: 35,
      reasons: ["Otto never called ask_question though the plan conflicts."],
    },
  ],
  checks: [{ name: "asks_question", passed: false, detail: "" }],
});

function detail(): EvalRunDetail {
  return {
    run: run("task-20261110-090000", "Nov 10, 09:00 UTC"),
    reviewers: [
      {
        reviewer: "claude:claude-opus-5-5",
        label: "claude-opus-5-5",
        stats: STATS,
      },
    ],
    categories: [
      { name: "must_ask", stats: { ...STATS, cases: 1, pass_rate: 0 } },
      { name: "simple_qa", stats: { ...STATS, cases: 1, pass_rate: 100 } },
    ],
    cases: [FAILED, evalCase({})],
    comparison: null,
    history: [],
    commands: [
      {
        label: "Repeat this run",
        command: "poetry run python -m x",
        cost: "Paid",
      },
    ],
  };
}

beforeEach(() => {
  server.use(
    getGetV2ListEvalRunsMockHandler200({
      runs: [
        run("task-20261110-090000", "Nov 10, 09:00 UTC"),
        run("task-baseline_native_run5", "Baseline native run 5"),
      ],
      commands: [],
    }),
    getGetV2GetEvalRunMockHandler200(detail()),
  );
});

describe("Admin evals page", () => {
  it("lists the saved runs and opens the newest", async () => {
    render(await EvalsPage());

    expect(
      await screen.findByRole("heading", {
        name: /Nov 10, 09:00 UTC/,
        hidden: true,
      }),
    ).toBeDefined();
    await userEvent.click(await screen.findByLabelText(/choose a run/i));
    expect(
      await screen.findByRole("menuitemradio", {
        name: /Baseline native run 5/,
        hidden: true,
      }),
    ).toBeDefined();
  });

  it("shows why a failed case failed", async () => {
    render(await EvalsPage());

    fireEvent.click(await screen.findByLabelText("must_ask_013: Fail"));

    const request = await screen.findByText("move nuvo to dev pls");
    const detail = request.closest('[aria-label^="Case:"]') as HTMLElement;
    expect(
      within(detail).getByText(
        "Otto never called ask_question though the plan conflicts.",
      ),
    ).toBeDefined();
    expect(within(detail).getByText("asks_question")).toBeDefined();
    expect(within(detail).getByText("Starts a task")).toBeDefined();
  });

  it("filters the cases by result", async () => {
    render(await EvalsPage());

    expect(await screen.findByLabelText("simple_qa_001: Pass")).toBeDefined();
    // `hidden: true` skips the visibility check: jsdom's getComputedStyle is
    // slow over the vendored block's large stylesheet.
    fireEvent.click(screen.getByRole("radio", { name: /^Fail/, hidden: true }));

    expect(screen.queryByLabelText("simple_qa_001: Pass")).toBeNull();
    expect(screen.getByLabelText("must_ask_013: Fail")).toBeDefined();
  });

  it("says when a run can't be found", async () => {
    server.use(
      http.get("*/api/proxy/api/admin/evals/runs/:runId", () =>
        HttpResponse.json(
          { detail: "No saved eval run has that id." },
          { status: 404 },
        ),
      ),
    );
    render(await EvalsPage());

    expect(await screen.findByText(/couldn't load this run/i)).toBeDefined();
  });
});
