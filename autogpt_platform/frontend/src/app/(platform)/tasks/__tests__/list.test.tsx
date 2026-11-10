import { getListTasksMockHandler401 } from "@/app/api/__generated__/endpoints/tasks/tasks.msw";
import { server } from "@/mocks/mock-server";
import { screen, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { MAYA, makeTask, renderTasksPage, serveTasks } from "./fixtures";

const TASKS = [
  makeTask({ id: "t1", title: "Write the launch post", status: "TODO" }),
  makeTask({
    id: "t2",
    title: "Research spring trends",
    status: "IN_PROGRESS",
    owner_expert_id: MAYA.id,
    started_at: new Date("2026-11-10T09:30:00Z"),
  }),
  makeTask({ id: "t3", title: "Pick a tone", status: "NEEDS_INPUT" }),
  makeTask({ id: "t4", title: "Pull churn numbers", status: "FAILED" }),
  makeTask({ id: "t5", title: "Send the newsletter", status: "DONE" }),
  makeTask({ id: "t6", title: "Old idea", status: "CANCELLED" }),
];

function rowOf(title: string) {
  const row = screen
    .getAllByTestId("task-row")
    .find((r) => within(r).queryByText(title) !== null);
  if (!row) throw new Error(`No row for ${title}`);
  return row;
}

function shownTitles() {
  return screen
    .getAllByTestId("task-row")
    .map((r) => r.querySelector("button")?.textContent ?? "");
}

describe("Tasks list", () => {
  test("shows each task with its status word and who works it", async () => {
    serveTasks(TASKS);
    renderTasksPage();

    expect(await screen.findByText("Write the launch post")).toBeDefined();

    expect(
      within(rowOf("Write the launch post")).getByText("To do"),
    ).toBeDefined();
    expect(
      within(rowOf("Write the launch post")).getByText("Otto"),
    ).toBeDefined();
    expect(
      within(rowOf("Research spring trends")).getByText("Working"),
    ).toBeDefined();
    expect(
      await within(rowOf("Research spring trends")).findByText("Maya"),
    ).toBeDefined();
    expect(within(rowOf("Pick a tone")).getByText("Needs you")).toBeDefined();
    expect(
      within(rowOf("Pull churn numbers")).getByText("Failed"),
    ).toBeDefined();
    expect(
      within(rowOf("Send the newsletter")).getByText("Done"),
    ).toBeDefined();
    expect(within(rowOf("Old idea")).getByText("Cancelled")).toBeDefined();
  });

  test("puts what needs you first", async () => {
    serveTasks(TASKS);
    renderTasksPage();

    await screen.findByText("Write the launch post");
    const titles = shownTitles();
    expect(titles[0]).toContain("Pick a tone");
    expect(titles[1]).toContain("Pull churn numbers");
    expect(titles.at(-1)).toContain("Old idea");
  });

  test("the status tabs count and filter the tasks", async () => {
    const user = userEvent.setup();
    serveTasks(TASKS);
    renderTasksPage();
    await screen.findByText("Write the launch post");

    const tabs = screen.getByRole("radiogroup", { name: "Show tasks" });
    const needsYou = within(tabs).getByRole("radio", { name: /^Needs you/ });
    expect(needsYou.textContent).toContain("2");

    await user.click(needsYou);

    expect(await screen.findByText("Pick a tone")).toBeDefined();
    expect(screen.getByText("Pull churn numbers")).toBeDefined();
    expect(screen.queryByText("Write the launch post")).toBeNull();
    expect(screen.queryByText("Send the newsletter")).toBeNull();

    await user.click(within(tabs).getByRole("radio", { name: /^Working/ }));

    expect(await screen.findByText("Write the launch post")).toBeDefined();
    expect(screen.getByText("Research spring trends")).toBeDefined();
    expect(screen.queryByText("Pick a tone")).toBeNull();
  });

  test("a new user is invited to hand off their first task", async () => {
    serveTasks([]);
    renderTasksPage();

    expect(await screen.findByText("Hand off your first task")).toBeDefined();
    expect(screen.getAllByRole("button", { name: "New task" })).toHaveLength(2);
  });

  test("an empty tab says so", async () => {
    const user = userEvent.setup();
    serveTasks([makeTask({ id: "t1", title: "Only one", status: "TODO" })]);
    renderTasksPage();
    await screen.findByText("Only one");

    await user.click(screen.getByRole("radio", { name: /^Cancelled/ }));

    expect(await screen.findByText("Nothing cancelled")).toBeDefined();
  });

  test("shows an error card when the tasks can't load", async () => {
    serveTasks([]);
    server.use(getListTasksMockHandler401());
    renderTasksPage();

    expect(
      await screen.findByText(/Your tasks could not be loaded/),
    ).toBeDefined();
  });
});
