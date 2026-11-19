import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import type { Task } from "@/app/api/__generated__/models/task";
import { render, screen, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TaskBoard } from "../../TaskBoard/TaskBoard";
import { TaskList } from "../TaskList";

const NOW = new Date(2026, 10, 17, 10, 0, 0).getTime();
const MIN = 60_000;

function task(id: string, title: string, extra: Partial<Task> = {}): Task {
  return { ...makeTask(id, title, "TODO"), ...extra };
}

const TASKS: Task[] = [
  task("done", "Send the invoice", { status: "DONE" }),
  task("working", "Draft the launch email", { status: "IN_PROGRESS" }),
  task("later", "Morning brief", { start_at: new Date(NOW + 30 * MIN) }),
  task("needs", "Pick a venue", { status: "NEEDS_INPUT" }),
  task("soon", "Check the inbox", { start_at: new Date(NOW + 5 * MIN) }),
];

function groupNames() {
  return screen
    .getAllByRole("region")
    .map((group) => group.getAttribute("aria-labelledby") ?? "")
    .map((id) => document.getElementById(id)?.textContent ?? "");
}

function titlesIn(name: string) {
  const group = screen.getByRole("region", { name: new RegExp(`^${name}`) });
  return within(group)
    .getAllByTestId("list-row")
    .map((row) => row.dataset.title);
}

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  vi.setSystemTime(NOW);
});

afterEach(() => {
  vi.useRealTimers();
});

describe("Task list", () => {
  it("groups tasks under the board's lanes, in the board's order, leaving out empty ones", () => {
    const board = render(
      <TaskBoard
        tasks={TASKS}
        experts={[]}
        selectedID={null}
        onOpen={vi.fn()}
      />,
    );
    const lanes = screen
      .getAllByTestId("board-lane")
      .filter((lane) => within(lane).queryAllByTestId("board-card").length > 0)
      .map((lane) => within(lane).getByRole("heading").textContent);
    board.unmount();

    render(
      <TaskList
        tasks={TASKS}
        experts={[]}
        selectedID={null}
        onOpen={vi.fn()}
      />,
    );

    expect(groupNames()).toEqual(lanes);
    expect(groupNames()).toEqual(["Needs you", "Scheduled", "Working", "Done"]);
    expect(titlesIn("Scheduled")).toEqual(["Check the inbox", "Morning brief"]);
    const working = screen.getByRole("region", { name: /^Working/ });
    expect(
      within(working).getByText("1", { selector: ".sr-only" }),
    ).toBeDefined();
  });

  it("opens a task when its row is clicked, and marks the selected one", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    const onOpen = vi.fn();
    render(
      <TaskList
        tasks={TASKS}
        experts={[]}
        selectedID="needs"
        onOpen={onOpen}
      />,
    );

    await user.click(screen.getByRole("button", { name: /^Draft the launch/ }));

    expect(onOpen).toHaveBeenCalledWith("working");
    expect(
      screen
        .getByRole("button", { name: /^Pick a venue/ })
        .getAttribute("aria-current"),
    ).toBe("true");
  });

  it("shows the board's empty line when there are no tasks", () => {
    render(
      <TaskList tasks={[]} experts={[]} selectedID={null} onOpen={vi.fn()} />,
    );

    expect(screen.getByText("Nothing here")).toBeDefined();
    expect(screen.queryAllByTestId("list-row")).toHaveLength(0);
  });
});
