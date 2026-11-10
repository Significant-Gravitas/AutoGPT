import { screen, waitFor, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { MAYA, makeTask, renderTasksPage, serveTasks } from "./fixtures";

function rowOf(title: string) {
  const row = screen
    .getAllByTestId("task-row")
    .find((r) => within(r).queryByText(title) !== null);
  if (!row) throw new Error(`No row for ${title}`);
  return row;
}

describe("Task actions", () => {
  test("creating a task posts it for the picked owner and shows it", async () => {
    const user = userEvent.setup();
    const api = serveTasks([makeTask({ id: "t1", title: "Existing task" })]);
    renderTasksPage();
    await screen.findByText("Existing task");

    await user.click(screen.getByRole("button", { name: "New task" }));
    const box = await screen.findByRole("textbox", {
      name: "What needs doing?",
    });
    await user.type(box, "Draft the sale email");
    await user.click(
      screen.getByRole("button", { name: "Owner: Otto. Change owner" }),
    );
    await user.click(
      await screen.findByRole("menuitemradio", { name: "Maya" }),
    );
    await user.click(screen.getByRole("button", { name: "Create task" }));

    await waitFor(() =>
      expect(api.created).toEqual([
        { title: "Draft the sale email", owner_expert_id: MAYA.id },
      ]),
    );
    expect(
      await screen.findByRole("heading", { name: "Draft the sale email" }),
    ).toBeDefined();
    expect(
      within(rowOf("Draft the sale email")).getByText("Maya"),
    ).toBeDefined();
  });

  test("the create button stays off until the task is described", async () => {
    const user = userEvent.setup();
    serveTasks([makeTask({ id: "t1", title: "Existing task" })]);
    renderTasksPage();
    await screen.findByText("Existing task");

    await user.click(screen.getByRole("button", { name: "New task" }));

    const create = await screen.findByRole("button", { name: "Create task" });
    expect(create.hasAttribute("disabled")).toBe(true);
  });

  test("renaming a task patches its title and shows the new one", async () => {
    const user = userEvent.setup();
    const api = serveTasks([makeTask({ id: "t1", title: "Old title" })]);
    renderTasksPage();
    await screen.findByText("Old title");

    await user.click(
      screen.getByRole("button", { name: "Actions for Old title" }),
    );
    await user.click(await screen.findByRole("menuitem", { name: "Rename" }));
    const field = await screen.findByRole("textbox", { name: "Task title" });
    await user.clear(field);
    await user.type(field, "New title{Enter}");

    await waitFor(() =>
      expect(api.updates).toEqual([{ id: "t1", body: { title: "New title" } }]),
    );
    expect(await screen.findByText("New title")).toBeDefined();
    expect(screen.queryByText("Old title")).toBeNull();
  });

  test("changing the status patches it and moves the pill", async () => {
    const user = userEvent.setup();
    const api = serveTasks([
      makeTask({ id: "t1", title: "Ship it", status: "IN_PROGRESS" }),
    ]);
    renderTasksPage();
    await screen.findByText("Ship it");

    await user.click(
      screen.getByRole("button", { name: "Actions for Ship it" }),
    );
    // The keyboard path: jsdom has no layout for the pointer's grace area
    // between a sub-trigger and its menu.
    (await screen.findByRole("menuitem", { name: "Change status" })).focus();
    await user.keyboard("{ArrowRight}");
    (await screen.findByRole("menuitemradio", { name: "Done" })).focus();
    await user.keyboard("{Enter}");

    await waitFor(() =>
      expect(api.updates).toEqual([{ id: "t1", body: { status: "DONE" } }]),
    );
    await waitFor(() =>
      expect(
        within(rowOf("Ship it")).getByTestId("task-status").dataset.status,
      ).toBe("DONE"),
    );
  });

  test("archiving asks first, then deletes the task and drops its row", async () => {
    const user = userEvent.setup();
    const api = serveTasks([
      makeTask({ id: "t1", title: "Keep me" }),
      makeTask({ id: "t2", title: "Archive me" }),
    ]);
    renderTasksPage();
    await screen.findByText("Archive me");

    await user.click(
      screen.getByRole("button", { name: "Actions for Archive me" }),
    );
    await user.click(await screen.findByRole("menuitem", { name: "Archive" }));
    await user.click(
      await screen.findByRole("button", { name: "Archive task" }),
    );

    await waitFor(() => expect(api.archived).toEqual(["t2"]));
    await waitFor(() => expect(screen.queryByText("Archive me")).toBeNull());
    expect(screen.getByText("Keep me")).toBeDefined();
  });
});
