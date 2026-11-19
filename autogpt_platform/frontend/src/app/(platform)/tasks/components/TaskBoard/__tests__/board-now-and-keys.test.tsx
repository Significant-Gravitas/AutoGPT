import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import type { Task } from "@/app/api/__generated__/models/task";
import type { TaskStatus } from "@/app/api/__generated__/models/taskStatus";
import { render, screen, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { TaskList } from "../../TaskList/TaskList";
import { TaskBoard } from "../TaskBoard";

function task(
  id: string,
  title: string,
  status: TaskStatus,
  extra: Partial<Task> = {},
): Task {
  return { ...makeTask(id, title, status), ...extra };
}

const NOW_TASKS: Task[] = [
  task("working", "Sort the inbox", "IN_PROGRESS", {
    status_text: "Reading the newest 40 emails",
    subtask_count: 5,
    subtasks_done: 3,
  }),
  task("done", "Book the venue", "DONE", {
    result: "## Summary\n**Booked** the hall for Friday\nMore below.",
  }),
  task("failed", "Find 25 companies", "FAILED", {
    failure_reason: "No web search was available.",
  }),
  task("todo", "Tidy the drive", "TODO"),
];

function card(name: RegExp) {
  return screen.getByRole("button", { name });
}

describe("Cards and rows say what is happening now", () => {
  it.each([
    ["board", TaskBoard],
    ["list", TaskList],
  ])(
    "shows the step, result or reason, and subtask progress (%s)",
    (_, View) => {
      render(
        <View
          tasks={NOW_TASKS}
          experts={[]}
          selectedID={null}
          onOpen={vi.fn()}
        />,
      );

      const working = within(card(/^Sort the inbox/));
      expect(working.getByText("Reading the newest 40 emails")).toBeDefined();
      expect(working.getByText("3 of 5 subtasks done")).toBeDefined();
      expect(
        within(card(/^Book the venue/)).getByText("Summary"),
      ).toBeDefined();
      expect(
        within(card(/^Find 25 companies/)).getByText(
          "No web search was available.",
        ),
      ).toBeDefined();
      expect(
        within(card(/^Tidy the drive/)).queryByTestId("task-now"),
      ).toBeNull();
    },
  );
});

const KEY_TASKS: Task[] = [
  task("a", "Answer the caterer", "NEEDS_INPUT"),
  task("b", "Draft the email", "IN_PROGRESS"),
  task("c", "Check the budget", "IN_PROGRESS"),
  task("d", "Send the invoice", "DONE"),
];

describe("Moving through the board with the keyboard", () => {
  it("moves with the arrows, says where each card is, and opens with Enter", async () => {
    const user = userEvent.setup();
    const onOpen = vi.fn();
    render(
      <TaskBoard
        tasks={KEY_TASKS}
        experts={[]}
        selectedID={null}
        onOpen={onOpen}
      />,
    );
    const tabbable = screen
      .getAllByTestId("board-card")
      .map((entry) => within(entry).getAllByRole("button")[0])
      .filter((button) => button.tabIndex === 0);
    expect(tabbable).toHaveLength(1);
    expect(
      within(card(/^Draft the email/)).getByText("Working, 1 of 2"),
    ).toBeDefined();

    await user.tab();
    expect(document.activeElement).toBe(card(/^Answer the caterer/));
    await user.keyboard("{ArrowRight}");
    expect(document.activeElement).toBe(card(/^Draft the email/));
    await user.keyboard("{ArrowDown}");
    expect(document.activeElement).toBe(card(/^Check the budget/));
    await user.keyboard("{ArrowRight}");
    expect(document.activeElement).toBe(card(/^Send the invoice/));
    await user.keyboard("{ArrowLeft}{ArrowLeft}");
    expect(document.activeElement).toBe(card(/^Answer the caterer/));

    await user.keyboard("{Enter}");
    expect(onOpen).toHaveBeenCalledWith("a");
  });

  it("keeps each card's ⋯ menu in reach of the keyboard and on screen", async () => {
    const user = userEvent.setup();
    render(
      <TaskBoard
        tasks={KEY_TASKS}
        experts={[]}
        selectedID={null}
        onOpen={vi.fn()}
      />,
    );

    await user.tab();
    await user.tab();

    const menu = screen.getByRole("button", {
      name: "Actions for Answer the caterer",
    });
    expect(document.activeElement).toBe(menu);
    expect(menu.className).not.toMatch(/opacity-0|invisible|hidden/);
  });

  it("moves up and down the list across its groups", async () => {
    const user = userEvent.setup();
    render(
      <TaskList
        tasks={KEY_TASKS}
        experts={[]}
        selectedID="c"
        onOpen={vi.fn()}
      />,
    );

    await user.tab();
    expect(document.activeElement).toBe(card(/^Check the budget/));
    await user.keyboard("{ArrowDown}");
    expect(document.activeElement).toBe(card(/^Send the invoice/));
    await user.keyboard("{Home}");
    expect(document.activeElement).toBe(card(/^Answer the caterer/));
    await user.keyboard("{ArrowDown}");
    expect(document.activeElement).toBe(card(/^Draft the email/));
  });
});
