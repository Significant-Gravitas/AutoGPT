import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { NewTaskDialog } from "../NewTaskDialog";

const mockToast = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/components/molecules/Toast/use-toast")
  >()),
  toast: mockToast,
}));

let calls: string[];

beforeEach(() => {
  calls = [];
  mockToast.mockClear();
  server.use(
    http.post("/api/proxy/api/tasks", () => {
      calls.push("create");
      return HttpResponse.json(makeTask("task-9", "Book the venue", "TODO"), {
        status: 201,
      });
    }),
    http.post("/api/proxy/api/tasks/task-9/run", () => {
      calls.push("run");
      return HttpResponse.json(
        makeTask("task-9", "Book the venue", "IN_PROGRESS"),
      );
    }),
    http.get("/api/proxy/api/tasks", () => HttpResponse.json({ tasks: [] })),
  );
});

function renderDialog(onCreated = vi.fn()) {
  render(
    <NewTaskDialog
      isOpen
      onOpenChange={vi.fn()}
      experts={[]}
      onCreated={onCreated}
    />,
  );
  return onCreated;
}

describe("New task dialog", () => {
  it("starts the task it creates and opens it", async () => {
    const onCreated = renderDialog();

    await userEvent.type(
      await screen.findByLabelText("What needs doing?"),
      "Book the venue",
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Create and start task" }),
    );

    await vi.waitFor(() => expect(onCreated).toHaveBeenCalledWith("task-9"));
    expect(calls).toEqual(["create", "run"]);
    expect(mockToast).toHaveBeenCalledWith({
      title: "Task started for Otto",
    });
  });

  it("saves a task for later without starting it", async () => {
    const onCreated = renderDialog();

    await userEvent.type(
      await screen.findByLabelText("What needs doing?"),
      "Book the venue",
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Save for later" }),
    );

    await vi.waitFor(() => expect(onCreated).toHaveBeenCalledWith("task-9"));
    expect(calls).toEqual(["create"]);
    expect(mockToast).toHaveBeenCalledWith({ title: "Task saved for Otto" });
  });

  it("keeps the task and says so when it could not start", async () => {
    server.use(
      http.post("/api/proxy/api/tasks/task-9/run", () =>
        HttpResponse.json({ detail: "Usage limit reached." }, { status: 429 }),
      ),
    );
    const onCreated = renderDialog();

    await userEvent.type(
      await screen.findByLabelText("What needs doing?"),
      "Book the venue{Enter}",
    );

    await vi.waitFor(() => expect(onCreated).toHaveBeenCalledWith("task-9"));
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Created the task, but couldn't start it",
        description: "Usage limit reached.",
      }),
    );
    expect(mockToast).toHaveBeenCalledWith({ title: "Task saved for Otto" });
  });
});
