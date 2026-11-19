import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import type { Task } from "@/app/api/__generated__/models/task";
import { render, screen, within } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TaskBoard } from "../TaskBoard";

const NOW = new Date(2026, 10, 15, 10, 0, 0).getTime();
const MIN = 60_000;

function task(id: string, title: string, extra: Partial<Task> = {}): Task {
  return { ...makeTask(id, title, "TODO"), ...extra };
}

function lane(name: string) {
  return screen.getByRole("region", { name: new RegExp(`^${name}`) });
}

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  vi.setSystemTime(NOW);
});

afterEach(() => {
  vi.useRealTimers();
});

describe("Task board with scheduled tasks", () => {
  it("puts scheduled tasks in their own lane, soonest first, with how soon they start", () => {
    render(
      <TaskBoard
        tasks={[
          task("later", "Write the digest", {
            start_at: new Date(NOW + 90 * MIN),
          }),
          task("plain", "Tidy the inbox"),
          task("soon", "Research AutoGPT", {
            start_at: new Date(NOW + 4 * MIN),
          }),
          task("missed", "Send the report", { missed_start: true }),
        ]}
        experts={[]}
        selectedID={null}
        onOpen={vi.fn()}
      />,
    );

    const scheduled = lane("Scheduled");
    const cards = within(scheduled).getAllByTestId("board-card");
    expect(cards.map((card) => card.textContent)).toEqual([
      expect.stringContaining("Research AutoGPT"),
      expect.stringContaining("Write the digest"),
    ]);
    const soon = cards[0];
    expect(within(soon).getByText("Scheduled")).toBeDefined();
    expect(within(soon).getByText("Starts in 4 min")).toBeDefined();
    expect(soon.dataset.schedule).toBe("scheduled");

    const missed = within(lane("Needs you")).getByTestId("board-card");
    expect(within(missed).getByText("Missed start")).toBeDefined();
    expect(missed.dataset.schedule).toBe("missed");

    const plain = within(lane("Working")).getByTestId("board-card");
    expect(within(plain).getByText("To do")).toBeDefined();
  });
});
