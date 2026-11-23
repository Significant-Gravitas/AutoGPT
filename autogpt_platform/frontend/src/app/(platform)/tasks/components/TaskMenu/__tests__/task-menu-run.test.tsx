import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import type { TaskStatus } from "@/app/api/__generated__/models/taskStatus";
import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
import { TaskMenu } from "../TaskMenu";

let calls: { url: string; body: unknown }[];

beforeEach(() => {
  calls = [];
  server.use(
    http.post("/api/proxy/api/tasks/task-1/run", async ({ request }) => {
      const text = await request.text();
      calls.push({ url: "run", body: text ? JSON.parse(text) : null });
      return HttpResponse.json(makeTask("task-1", "Plan", "IN_PROGRESS"));
    }),
    http.post("/api/proxy/api/tasks/task-1/cancel", () => {
      calls.push({ url: "cancel", body: null });
      return HttpResponse.json(makeTask("task-1", "Plan", "CANCELLED"));
    }),
  );
});

async function openMenu(status: TaskStatus) {
  render(
    <TaskMenu task={makeTask("task-1", "Plan", status)} onRename={vi.fn()} />,
  );
  await userEvent.click(
    screen.getByRole("button", { name: "Actions for Plan" }),
  );
}

describe("Task menu run actions", () => {
  it("tries a cancelled task again with a note", async () => {
    await openMenu("CANCELLED");

    await userEvent.click(
      await screen.findByRole("menuitem", { name: "Try again" }),
    );
    const form = await screen.findByRole("form", {
      name: "Try the task again",
    });
    await userEvent.type(
      within(form).getByLabelText("Anything to do differently?"),
      "Shorter this time",
    );
    await userEvent.click(
      within(form).getByRole("button", { name: "Try again" }),
    );

    await vi.waitFor(() =>
      expect(calls).toEqual([
        { url: "run", body: { instructions: "Shorter this time" } },
      ]),
    );
  });

  it("stops a running task once confirmed", async () => {
    await openMenu("NEEDS_INPUT");

    await userEvent.click(
      await screen.findByRole("menuitem", { name: "Stop" }),
    );
    const dialog = await screen.findByRole("dialog", {
      name: "Stop this task?",
    });
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Stop task" }),
    );

    await vi.waitFor(() =>
      expect(calls).toEqual([{ url: "cancel", body: null }]),
    );
  });

  it("starts a task that is to do at once", async () => {
    await openMenu("TODO");

    await userEvent.click(
      await screen.findByRole("menuitem", { name: "Start" }),
    );

    await vi.waitFor(() => expect(calls).toEqual([{ url: "run", body: null }]));
  });
});
