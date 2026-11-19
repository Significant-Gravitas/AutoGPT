import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import type { Task } from "@/app/api/__generated__/models/task";
import { render, screen, within } from "@/tests/integrations/test-utils";
import { describe, expect, it, vi } from "vitest";
import { TaskList } from "../../TaskList/TaskList";
import { TaskBoard } from "../TaskBoard";

const QUEUED: Task = {
  ...makeTask("queued", "Write the weekly update", "IN_PROGRESS"),
  is_queued: true,
  status_text: "Queued: starts when a slot frees",
};

const FAILED: Task = {
  ...makeTask("failed", "Find 25 companies", "FAILED"),
  failure_reason: "Not done: 0 of 25 prospects listed, no web search.",
  result: "Not done: 0 of 25 prospects listed.",
};

describe("A queued task", () => {
  it.each([
    ["board", TaskBoard],
    ["list", TaskList],
  ])(
    "reads Queued in the Working lane, starting when a slot frees (%s)",
    (_, View) => {
      render(
        <View
          tasks={[QUEUED, FAILED]}
          experts={[]}
          selectedID={null}
          onOpen={vi.fn()}
        />,
      );

      const working = screen.getByRole("region", { name: /^Working/ });
      const card = within(working).getByRole("button", {
        name: /^Write the weekly update/,
      });
      const pill = within(card).getByTestId("task-status");
      expect(pill.textContent).toBe("Queued");
      expect(within(card).getByText("Starts when a slot frees")).toBeDefined();
      expect(within(card).getByTestId("task-state-icon").dataset.queued).toBe(
        "true",
      );

      const failed = screen.getByRole("button", { name: /^Find 25 companies/ });
      expect(within(failed).getByTestId("task-status").textContent).toBe(
        "Failed",
      );
      expect(
        within(failed).getByText(
          "Not done: 0 of 25 prospects listed, no web search.",
        ),
      ).toBeDefined();
    },
  );
});
