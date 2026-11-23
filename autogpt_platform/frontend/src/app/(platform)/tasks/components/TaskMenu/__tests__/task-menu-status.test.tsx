import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import type { TaskStatus } from "@/app/api/__generated__/models/taskStatus";
import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { TaskMenu } from "../TaskMenu";

async function openMenu(status: TaskStatus) {
  render(
    <TaskMenu task={makeTask("task-1", "Plan", status)} onRename={vi.fn()} />,
  );
  await userEvent.click(
    screen.getByRole("button", { name: "Actions for Plan" }),
  );
}

// Radix submenus don't open on click in jsdom: focus, then ArrowRight.
async function openStatusMenu() {
  const trigger = await screen.findByRole("menuitem", {
    name: "Change status",
  });
  trigger.focus();
  await userEvent.keyboard("{ArrowRight}");
  return screen.findAllByRole("menuitemradio");
}

describe("Task menu status choices", () => {
  it("offers only what a person can set: reopen, done or cancelled", async () => {
    await openMenu("IN_PROGRESS");

    const items = await openStatusMenu();

    expect(items.map((item) => item.textContent)).toEqual([
      "To do",
      "Done",
      "Cancelled",
    ]);
  });

  it("offers Try again instead of a status for a failed task", async () => {
    await openMenu("FAILED");

    expect(
      await screen.findByRole("menuitem", { name: "Try again" }),
    ).toBeDefined();
    expect(screen.queryByRole("menuitem", { name: "Change status" })).toBe(
      null,
    );
  });
});
