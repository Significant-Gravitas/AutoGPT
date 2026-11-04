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
import { TasksWorkspace } from "../components/TasksWorkspace/TasksWorkspace";
import { formatAge, matchesAssignees, statusOf } from "../data/helpers";
import type { Task } from "../data/types";
import { expertChoices } from "../helpers";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
  usePathname: () => "/tasks",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

const mockToast = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/components/molecules/Toast/use-toast")
  >()),
  toast: mockToast,
}));

const minutesAgo = (n: number) =>
  new Date(Date.now() - n * 60_000).toISOString();

function row(id: string, number: number, extra: Record<string, unknown>) {
  return {
    id,
    number,
    title: `Task ${number}`,
    description: "",
    status: "in_progress",
    priority: "none",
    assignee: { type: "otto", id: null, name: "Otto" },
    labels: [],
    subtask_counts: { total: 0, done: 0 },
    metrics: { tool_calls: 2, tokens: 9000, cost_usd: 0.03, duration_ms: 0 },
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

const DONE = row("d1", 11, {
  title: "Pick a project tool",
  status: "done",
  raw_status: "done",
  state: "done",
  now: "1 of 1 checks met",
  outcome_summary: "Linear fits the design team best.",
  completed_at: minutesAgo(120),
});

const RUNNING = row("r1", 12, {
  title: "Research competitors",
  now: "Reading competitor sites",
});

function serveTasks(initial: unknown[]) {
  let tasks = [...initial];
  const urls: string[] = [];
  server.use(
    http.get("*/api/tasks/experts", () => HttpResponse.json({ experts: [] })),
    http.get("*/api/tasks", ({ request }) => {
      urls.push(request.url);
      return HttpResponse.json({ tasks });
    }),
  );
  return {
    urls,
    remove(id: string) {
      tasks = tasks.filter((t) => (t as { id: string }).id !== id);
    },
  };
}

async function findRow(title: string) {
  return (await screen.findByText(title)).closest(
    "[data-testid='inbox-row']",
  ) as HTMLElement;
}

beforeEach(() => mockToast.mockReset());

describe("task cards", () => {
  test("a done card shows its result, never checks, cost or tokens", async () => {
    serveTasks([DONE, RUNNING]);
    render(<TasksWorkspace />);

    const done = await findRow("Pick a project tool");
    expect(
      within(done).getByText("Linear fits the design team best."),
    ).toBeTruthy();
    expect(within(done).getByTestId("task-status").textContent).toBe("Done");
    expect(within(done).getByText("· 2h ago")).toBeTruthy();

    expect(screen.queryByText(/checks met/)).toBeNull();
    expect(screen.queryByText(/tokens|tool call|\$0\.03/)).toBeNull();
  });

  test("a working card shows only its live step and how long ago", async () => {
    serveTasks([RUNNING]);
    render(<TasksWorkspace />);

    const card = await findRow("Research competitors");
    expect(within(card).getByText("Reading competitor sites")).toBeTruthy();
    expect(within(card).getByText("· 1m ago")).toBeTruthy();
    expect(within(card).getAllByTestId("task-status")).toHaveLength(1);
  });
});

describe("rename and delete", () => {
  test("rename shows the new title at once and rolls back when it fails", async () => {
    serveTasks([RUNNING]);
    let release: () => void = () => {};
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    server.use(
      http.patch("*/api/tasks/r1", async () => {
        await gate;
        return HttpResponse.json({ detail: "nope" }, { status: 500 });
      }),
    );
    const user = userEvent.setup();
    render(<TasksWorkspace />);

    const card = await findRow("Research competitors");
    await user.click(
      within(card).getByRole("button", { name: "Task actions" }),
    );
    await user.click(await screen.findByRole("menuitem", { name: "Rename" }));
    const field = await screen.findByRole("textbox", { name: "Task title" });
    await user.clear(field);
    await user.type(field, "Research five competitors{Enter}");

    expect(await screen.findByText("Research five competitors")).toBeTruthy();

    release();
    expect(await screen.findByText("Research competitors")).toBeTruthy();
    await waitFor(() =>
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Couldn't rename the task" }),
      ),
    );
  });

  test("delete asks first, then removes the row at once", async () => {
    const api = serveTasks([DONE, RUNNING]);
    const deleted: string[] = [];
    server.use(
      http.delete("*/api/tasks/:id", ({ params }) => {
        deleted.push(String(params.id));
        api.remove(String(params.id));
        return new HttpResponse(null, { status: 204 });
      }),
    );
    const user = userEvent.setup();
    render(<TasksWorkspace />);

    const card = await findRow("Research competitors");
    await user.click(
      within(card).getByRole("button", { name: "Task actions" }),
    );
    await user.click(await screen.findByRole("menuitem", { name: "Delete" }));
    expect(deleted).toHaveLength(0);

    await user.click(
      await screen.findByRole("button", { name: "Delete task" }),
    );

    await waitFor(() =>
      expect(screen.queryByText("Research competitors")).toBeNull(),
    );
    expect(screen.getByText("Pick a project tool")).toBeTruthy();
    await waitFor(() => expect(deleted).toEqual(["r1"]));
  });
});

describe("filters", () => {
  test("Mine asks the server for my tasks", async () => {
    const api = serveTasks([RUNNING]);
    const user = userEvent.setup();
    render(<TasksWorkspace />);

    await screen.findByText("Research competitors");
    await user.click(screen.getByRole("button", { name: "Filter tasks" }));
    await user.click(
      await screen.findByRole("menuitemcheckbox", { name: "Mine" }),
    );

    await waitFor(() =>
      expect(api.urls.some((u) => u.includes("mine=true"))).toBe(true),
    );
    // The last list stays on screen while the filtered one loads.
    expect(screen.getByText("Research competitors")).toBeTruthy();
  });

  test("the Expert filter offers Otto and matches Otto's tasks", () => {
    expect(expertChoices([{ id: "e1", name: "Maya", busy: false }])).toEqual([
      { id: "otto", name: "Otto" },
      { id: "e1", name: "Maya" },
    ]);
    const otto = { assignee: { type: "otto", id: null, name: "Otto" } };
    const maya = { assignee: { type: "expert", id: "e1", name: "Maya" } };
    expect(matchesAssignees(otto as Task, ["otto"])).toBe(true);
    expect(matchesAssignees(maya as Task, ["otto"])).toBe(false);
    expect(matchesAssignees(maya as Task, ["e1"])).toBe(true);
  });
});

describe("status vocabulary", () => {
  test("every state lands in exactly one of the four buckets", () => {
    expect(
      (["needs_you", "stuck", "working", "done", "stopped"] as const).map(
        (s) => [statusOf(s).bucket, statusOf(s).word],
      ),
    ).toEqual([
      ["needs_you", "Needs you"],
      ["needs_you", "Failed"],
      ["working", "Working"],
      ["done", "Done"],
      ["cancelled", "Cancelled"],
    ]);
  });

  test("durations read short, never minutes and seconds", () => {
    expect(formatAge(20_000)).toBe("20s");
    expect(formatAge(344 * 60_000 + 6_000)).toBe("5h");
    expect(formatAge(2 * 86_400_000 + 5_000)).toBe("2d");
  });
});
