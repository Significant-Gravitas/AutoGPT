import { getGetTaskMockHandler401 } from "@/app/api/__generated__/endpoints/tasks/tasks.msw";
import { server } from "@/mocks/mock-server";
import { screen, waitFor, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { MAYA, makeTask, renderTasksPage, serveTasks } from "./fixtures";

const TASK = makeTask({
  id: "t1",
  title: "Research spring trends",
  description: "Look at what sold last spring\nand what people search for.",
  status: "DONE",
  owner_expert_id: MAYA.id,
  created_at: new Date("2026-11-08T09:00:00Z"),
  started_at: new Date("2026-11-08T09:05:00Z"),
  completed_at: new Date("2026-11-08T10:00:00Z"),
  session_ids: ["chat-a", "chat-b"],
});

async function openTask(title: string) {
  const user = userEvent.setup();
  await user.click(await screen.findByText(title));
  return user;
}

function panel() {
  return screen.getByRole("complementary", { name: "Task details" });
}

describe("Task panel", () => {
  test("shows the task's title, status, owner, description and times", async () => {
    serveTasks([TASK]);
    renderTasksPage();

    await openTask("Research spring trends");

    const heading = await within(panel()).findByRole("heading", {
      name: "Research spring trends",
    });
    await waitFor(() => expect(document.activeElement).toBe(heading));
    expect(within(panel()).getByTestId("task-status").dataset.status).toBe(
      "DONE",
    );
    expect(within(panel()).getByText("Maya")).toBeDefined();
    expect(
      within(panel()).getByText(/Look at what sold last spring/),
    ).toBeDefined();

    const times = within(panel()).getByRole("region", { name: "Timeline" });
    expect(within(times).getByText("Created")).toBeDefined();
    expect(within(times).getByText("Started")).toBeDefined();
    expect(within(times).getByText("Finished")).toBeDefined();
    expect(within(times).queryByText("Not yet")).toBeNull();
  });

  test("links to every chat the task was worked in", async () => {
    serveTasks([TASK]);
    renderTasksPage();

    await openTask("Research spring trends");

    const chats = await within(panel()).findByRole("region", { name: "Chats" });
    const links = within(chats).getAllByRole("link");
    expect(links.map((l) => l.getAttribute("href"))).toEqual([
      "/copilot?sessionId=chat-a",
      "/copilot?sessionId=chat-b",
    ]);
    expect(links.map((l) => l.textContent)).toEqual(["Chat 1", "Chat 2"]);
  });

  test("a task not started yet says so and has no chats", async () => {
    serveTasks([
      makeTask({ id: "t2", title: "Draft the sale email", status: "TODO" }),
    ]);
    renderTasksPage();

    await openTask("Draft the sale email");

    const times = await within(panel()).findByRole("region", {
      name: "Timeline",
    });
    expect(within(times).getAllByText("Not yet")).toHaveLength(2);
    expect(within(panel()).getByText("No chats yet")).toBeDefined();
    expect(within(panel()).getByText("No description")).toBeDefined();
    expect(within(panel()).getByText("Otto")).toBeDefined();
  });

  test("Escape closes the panel and puts focus back on the row", async () => {
    serveTasks([TASK]);
    renderTasksPage();

    const user = await openTask("Research spring trends");
    await within(panel()).findByRole("heading", {
      name: "Research spring trends",
    });

    await user.keyboard("{Escape}");

    await waitFor(() =>
      expect(
        screen.queryByRole("complementary", { name: "Task details" }),
      ).toBeNull(),
    );
    await waitFor(() => expect(document.activeElement?.id).toBe("task-row-t1"));
  });

  test("shows an error card when the task can't load", async () => {
    serveTasks([TASK]);
    server.use(getGetTaskMockHandler401());
    renderTasksPage();

    await openTask("Research spring trends");

    expect(
      await within(panel()).findByText(/This task could not be loaded/),
    ).toBeDefined();
  });
});
