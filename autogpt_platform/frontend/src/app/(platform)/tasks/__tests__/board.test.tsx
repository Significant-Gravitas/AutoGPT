import { screen, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { makeTask, renderTasksPage, serveTasks } from "./fixtures";

const TASKS = [
  makeTask({ id: "t1", title: "Write the launch post", status: "TODO" }),
  makeTask({ id: "t2", title: "Research trends", status: "IN_PROGRESS" }),
  makeTask({ id: "t3", title: "Pick a tone", status: "NEEDS_INPUT" }),
  makeTask({ id: "t4", title: "Pull churn numbers", status: "FAILED" }),
  makeTask({ id: "t5", title: "Send the newsletter", status: "DONE" }),
  makeTask({ id: "t6", title: "Old idea", status: "CANCELLED" }),
];

function laneTitles(name: string) {
  const lane = screen.getByRole("region", { name });
  return within(lane)
    .getAllByTestId("board-card")
    .map((card) => card.querySelector("button")?.textContent ?? "");
}

describe("Tasks board", () => {
  test("groups the tasks into four lanes by status", async () => {
    const user = userEvent.setup();
    serveTasks(TASKS);
    renderTasksPage();
    await screen.findByText("Write the launch post");

    await user.click(screen.getByRole("radio", { name: "Board" }));

    expect(
      await screen.findByRole("region", { name: "Needs you" }),
    ).toBeDefined();
    const needsYou = laneTitles("Needs you");
    expect(needsYou).toHaveLength(2);
    expect(needsYou.join()).toContain("Pick a tone");
    expect(needsYou.join()).toContain("Pull churn numbers");

    const working = laneTitles("Working");
    expect(working).toHaveLength(2);
    expect(working.join()).toContain("Write the launch post");
    expect(working.join()).toContain("Research trends");

    expect(laneTitles("Done").join()).toContain("Send the newsletter");
    expect(laneTitles("Cancelled").join()).toContain("Old idea");
  });

  test("an empty lane says nothing is there", async () => {
    const user = userEvent.setup();
    serveTasks([TASKS[0]]);
    renderTasksPage();
    await screen.findByText("Write the launch post");

    await user.click(screen.getByRole("radio", { name: "Board" }));

    const done = await screen.findByRole("region", { name: "Done" });
    expect(within(done).getByText("Nothing here")).toBeDefined();
  });
});
