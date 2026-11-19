import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import type { Task } from "@/app/api/__generated__/models/task";
import { render, screen, within } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TaskBoard } from "../TaskBoard";

const NOW = new Date(2026, 10, 17, 10, 0, 0).getTime();
const MIN = 60_000;
const WATCH = {
  webhook_id: "wh-1",
  provider: "github",
  trigger_block_id: "6c60ec01-8128-419e-988f-96a063ee2fea",
  line: "PR opened on acme/web",
};

function task(id: string, title: string, extra: Partial<Task> = {}): Task {
  return { ...makeTask(id, title, "TODO"), ...extra };
}

function lane(name: string) {
  return screen.getByRole("region", { name: new RegExp(`^${name}`) });
}

function renderBoard(tasks: Task[]) {
  render(
    <TaskBoard tasks={tasks} experts={[]} selectedID={null} onOpen={vi.fn()} />,
  );
}

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  vi.setSystemTime(NOW);
});

afterEach(() => {
  vi.useRealTimers();
});

describe("Task board with watching tasks", () => {
  it("puts tasks waiting for an event in a Watching lane, repeating ones stay in Scheduled", () => {
    renderBoard([
      task("watch", "Review new PRs", {
        repeat: { label: WATCH.line, watch: WATCH },
      }),
      task("daily", "Morning brief", {
        start_at: new Date(NOW + 4 * MIN),
        repeat: { crons: ["0 8 * * *"], label: "Every day at 08:00" },
      }),
    ]);

    const watching = within(lane("Watching")).getByTestId("board-card");
    expect(within(watching).getByText("Review new PRs")).toBeDefined();
    expect(within(watching).getByText("Watching")).toBeDefined();
    expect(within(watching).getByText("PR opened on acme/web")).toBeDefined();
    expect(watching.dataset.schedule).toBe("watching");

    const scheduled = within(lane("Scheduled")).getByTestId("board-card");
    expect(within(scheduled).getByText("Morning brief")).toBeDefined();
  });

  it("keeps a paused watching task in Watching, and a running one in Working", () => {
    renderBoard([
      task("paused", "Review new PRs", {
        repeat: { label: WATCH.line, paused: true, watch: WATCH },
      }),
      task("running", "Triage issues", {
        status: "IN_PROGRESS",
        repeat: { label: "Issue opened on acme/web", watch: WATCH },
      }),
    ]);

    const paused = within(lane("Watching")).getByTestId("board-card");
    expect(
      within(paused).getByText("PR opened on acme/web"),
    ).toBeDefined();
    expect(paused.dataset.schedule).toBe("paused");
    expect(within(lane("Working")).getByText("Triage issues")).toBeDefined();
  });

  it("shows no Watching lane when nothing watches", () => {
    renderBoard([task("plain", "Tidy the inbox")]);

    expect(screen.queryByRole("region", { name: /^Watching/ })).toBeNull();
  });
});
