import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import { server } from "@/mocks/mock-server";
import {
  act,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { NewTaskDialog } from "../NewTaskDialog";

let creates: number;

beforeEach(() => {
  creates = 0;
  window.sessionStorage.clear();
  server.use(
    http.post("/api/proxy/api/tasks", async () => {
      creates += 1;
      await new Promise((resolve) => setTimeout(resolve, 40));
      return HttpResponse.json(makeTask("task-9", "Book the venue", "TODO"), {
        status: 201,
      });
    }),
    http.post("/api/proxy/api/tasks/task-9/run", () =>
      HttpResponse.json(makeTask("task-9", "Book the venue", "IN_PROGRESS")),
    ),
    http.get("/api/proxy/api/tasks", () => HttpResponse.json({ tasks: [] })),
  );
});

async function typeTask() {
  const onCreated = vi.fn();
  render(
    <NewTaskDialog
      isOpen
      onOpenChange={vi.fn()}
      experts={[]}
      onCreated={onCreated}
    />,
  );
  await userEvent.type(
    await screen.findByLabelText("What needs doing?"),
    "Book the venue",
  );
  return onCreated;
}

/** Two clicks in the same tick, before React can re-render: what a fast
 *  double click does to a button whose disabled state lags a render. */
function clickTwice(name: string) {
  const button = screen.getByRole("button", { name });
  act(() => {
    button.click();
    button.click();
  });
}

async function settle(onCreated: ReturnType<typeof vi.fn>) {
  await vi.waitFor(() => expect(onCreated).toHaveBeenCalled());
  await new Promise((resolve) => setTimeout(resolve, 100));
}

describe("New task: one task per submit", () => {
  it("creates one task on a double click of Save for later", async () => {
    const onCreated = await typeTask();

    clickTwice("Save for later");

    await settle(onCreated);
    expect(creates).toBe(1);
  });

  it("creates one task on a double click of Create and start", async () => {
    const onCreated = await typeTask();

    clickTwice("Create and start task");

    await settle(onCreated);
    expect(creates).toBe(1);
  });

  it("creates one task when Enter is pressed twice", async () => {
    const onCreated = await typeTask();

    const field = screen.getByLabelText("What needs doing?");
    act(() => {
      fireEvent.keyDown(field, { key: "Enter" });
      fireEvent.keyDown(field, { key: "Enter" });
    });

    await settle(onCreated);
    expect(creates).toBe(1);
  });
});
