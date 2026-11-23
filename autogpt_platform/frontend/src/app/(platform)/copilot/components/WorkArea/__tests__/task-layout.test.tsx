import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import type { Task } from "@/app/api/__generated__/models/task";
import type { TaskArtifact } from "@/app/api/__generated__/models/taskArtifact";
import type { TaskRun } from "@/app/api/__generated__/models/taskRun";
import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
import { TaskTab } from "../TaskTab";

const TASK_URL = "/api/proxy/api/tasks/task-1";

let task: Task;
let artifacts: TaskArtifact[];
let runs: TaskRun[];

function serve() {
  server.use(
    http.get(TASK_URL, () => HttpResponse.json(task)),
    http.get(`${TASK_URL}/record`, () =>
      HttpResponse.json({
        task,
        artifacts,
        runs,
        evidence: [],
        activity: [
          {
            id: "l-1",
            task_id: "task-1",
            created_at: new Date("2026-11-19T09:01:00Z"),
            actor_type: "EXPERT",
            kind: "PROGRESS",
            text: "Searching for partners",
            is_milestone: true,
            data: {},
          },
        ],
      }),
    ),
  );
}

function regionOrder() {
  return screen
    .getAllByRole("region")
    .map((region) => region.getAttribute("aria-label"));
}

function expectInOrder(names: string[]) {
  const order = regionOrder();
  const at = names.map((name) => order.indexOf(name));
  expect(
    at.every((index) => index >= 0),
    order.join(", "),
  ).toBe(true);
  expect([...at].sort((a, b) => a - b)).toEqual(at);
}

function renderTab() {
  return render(
    <TaskTab
      taskID="task-1"
      sessionID={null}
      experts={[]}
      onOpenArtifact={vi.fn()}
    />,
  );
}

const DONE = {
  ...makeTask("task-1", "Plan the launch", "DONE"),
  description: "Keep it friendly.",
  result: "Found 25 partners.\nSee the list.",
};

beforeEach(() => {
  artifacts = [];
  runs = [];
  serve();
});

afterEach(() => vi.restoreAllMocks());

describe("Task tab layout", () => {
  it("leads a finished task with what happened, About folded", async () => {
    task = DONE;
    renderTab();
    await screen.findByRole("region", { name: "Activity" });

    expectInOrder(["Status", "Result", "About", "Details", "Activity"]);
    const status = screen.getByRole("region", { name: "Status" });
    expect(within(status).getByText("Found 25 partners.")).toBeDefined();
    const about = screen.getByRole("region", { name: "About" });
    expect(within(about).queryByRole("button", { name: /^Goal/ })).toBeNull();

    await userEvent.click(
      within(about).getByRole("button", { name: /Keep it friendly/ }),
    );
    expect(within(about).getByRole("button", { name: /^Goal/ })).toBeDefined();
  });

  it("keeps About near the top while a new task has no result", async () => {
    task = {
      ...makeTask("task-1", "Plan the launch", "TODO"),
      description: "Keep it friendly.",
    };
    renderTab();
    await screen.findByRole("region", { name: "Activity" });

    expectInOrder(["Status", "About", "Details", "Activity"]);
    expect(screen.queryByRole("region", { name: "Result" })).toBeNull();
    const about = screen.getByRole("region", { name: "About" });
    expect(
      within(about).getByRole("button", { name: /^Description/ }),
    ).toBeDefined();
  });
});

describe("Taking the result away", () => {
  it("copies the result's words", async () => {
    const user = userEvent.setup();
    const write = vi.spyOn(navigator.clipboard, "writeText");
    task = DONE;
    renderTab();
    const result = await screen.findByRole("region", { name: "Result" });

    await user.click(within(result).getByRole("button", { name: "Copy" }));

    expect(write).toHaveBeenCalledWith("Found 25 partners.\nSee the list.");
    expect(
      await within(result).findByRole("button", { name: "Copied" }),
    ).toBeDefined();
  });

  it("downloads the result as Markdown named after the task", async () => {
    const saved: { download: string; href: string }[] = [];
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (
      this: HTMLAnchorElement,
    ) {
      saved.push({ download: this.download, href: this.href });
    });
    URL.createObjectURL = vi.fn(() => "blob:result");
    URL.revokeObjectURL = vi.fn();
    task = DONE;
    renderTab();
    const result = await screen.findByRole("region", { name: "Result" });

    await userEvent.click(
      within(result).getByRole("button", { name: "Download" }),
    );

    expect(saved).toEqual([
      { download: "Plan the launch.md", href: "blob:result" },
    ]);
    expect(
      await within(result).findByRole("button", { name: "Saved" }),
    ).toBeDefined();
  });

  it("downloads the result's own document when it has one", async () => {
    const saved: { download: string; href: string }[] = [];
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (
      this: HTMLAnchorElement,
    ) {
      saved.push({ download: this.download, href: this.href });
    });
    task = DONE;
    artifacts = [
      {
        id: "art-1",
        task_id: "task-1",
        created_at: new Date("2026-11-19T09:05:00Z"),
        kind: "DOCUMENT",
        title: "Partners.md",
        workspace_file_id: "file-1",
      },
    ];
    renderTab();
    const result = await screen.findByRole("region", { name: "Result" });

    await userEvent.click(
      within(result).getByRole("button", { name: "Download" }),
    );

    expect(saved).toHaveLength(1);
    expect(saved[0].download).toBe("Partners.md");
    expect(saved[0].href).toContain(
      "/api/proxy/api/workspace/files/file-1/download",
    );
  });

  it("copies a link that opens the task", async () => {
    const user = userEvent.setup();
    const write = vi.spyOn(navigator.clipboard, "writeText");
    task = DONE;
    renderTab();

    await user.click(await screen.findByRole("button", { name: "Copy link" }));

    expect(write).toHaveBeenCalledWith(
      `${window.location.origin}/home?show=tasks&task=task-1`,
    );
    expect(
      await screen.findByRole("button", { name: "Link copied" }),
    ).toBeDefined();
  });
});

describe("Runs of a repeating task", () => {
  it("lists each run, a failed one marked, and says when several failed in a row", async () => {
    task = {
      ...makeTask("task-1", "Morning digest", "TODO"),
      repeat: {
        crons: ["0 8 * * *"],
        timezone: "UTC",
        label: "Every day at 08:00",
        failed_in_a_row: 2,
      },
    };
    runs = [
      {
        chat_session_id: "run-3",
        started_at: new Date("2026-11-19T08:00:00Z"),
        outcome: "failed",
        failure_reason: "No web search was available.",
      },
      {
        chat_session_id: "run-2",
        started_at: new Date("2026-11-18T08:00:00Z"),
        outcome: "failed",
        failure_reason: "The site was down.",
      },
      {
        chat_session_id: "run-1",
        started_at: new Date("2026-11-17T08:00:00Z"),
        outcome: "done",
        result: "Five stories.",
      },
    ];
    renderTab();

    const block = await screen.findByRole("region", { name: "Runs" });
    expect(within(block).getByText("2 failed in a row")).toBeDefined();
    expect(within(block).getAllByText("Failed")).toHaveLength(2);
    expect(
      within(block).getByText("No web search was available."),
    ).toBeDefined();
    expect(within(block).getByText("Done")).toBeDefined();
  });
});
