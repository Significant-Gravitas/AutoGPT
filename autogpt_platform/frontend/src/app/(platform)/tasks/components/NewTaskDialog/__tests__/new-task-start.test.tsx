import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
import { useListTasks } from "@/app/api/__generated__/endpoints/tasks/tasks";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { okData } from "@/app/api/helpers";
import { TaskList } from "../../TaskList/TaskList";
import { useState } from "react";
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
    await vi.waitFor(() =>
      expect(mockToast).toHaveBeenCalledWith({
        title: "Task started for Otto",
      }),
    );
    expect(calls).toEqual(["create", "run"]);
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

  it("keeps the task and says so once, with a retry, when it could not start", async () => {
    let runs = 0;
    server.use(
      http.post("/api/proxy/api/tasks/task-9/run", () => {
        runs += 1;
        return runs === 1
          ? HttpResponse.json(
              { detail: "Usage limit reached." },
              { status: 429 },
            )
          : HttpResponse.json(
              makeTask("task-9", "Book the venue", "IN_PROGRESS"),
            );
      }),
    );
    const onCreated = renderDialog();

    await userEvent.type(
      await screen.findByLabelText("What needs doing?"),
      "Book the venue{Enter}",
    );

    await vi.waitFor(() => expect(onCreated).toHaveBeenCalledWith("task-9"));
    await vi.waitFor(() => expect(mockToast).toHaveBeenCalledTimes(1));
    const [failure] = mockToast.mock.calls[0];
    expect(failure).toEqual(
      expect.objectContaining({
        title: "Saved for Otto, but couldn't start",
        description: "Usage limit reached.",
        variant: "destructive",
      }),
    );

    // The closing dialog still holds pointer-events off the page.
    render(<>{failure.action}</>);
    await userEvent
      .setup({ pointerEventsCheck: 0 })
      .click(screen.getByRole("button", { name: "Retry" }));

    await vi.waitFor(() => expect(runs).toBe(2));
    await vi.waitFor(() =>
      expect(mockToast).toHaveBeenLastCalledWith({
        title: "Task started for Otto",
      }),
    );
  });

  it("puts the new task on the board before the server answers", async () => {
    const held: { answer: (() => void) | null } = { answer: null };
    const created = makeTask("task-9", "Book the venue", "TODO");
    let saved: typeof created | null = null;
    server.use(
      http.post("/api/proxy/api/tasks", async () => {
        await new Promise<void>((resolve) => {
          held.answer = resolve;
        });
        saved = created;
        return HttpResponse.json(created, { status: 201 });
      }),
      http.get("/api/proxy/api/tasks", () =>
        HttpResponse.json({ tasks: saved ? [saved] : [] }),
      ),
    );
    render(
      <>
        <BoardHarness />
        <NewTaskDialog isOpen onOpenChange={vi.fn()} experts={[]} />
      </>,
    );
    await screen.findByText("Nothing here");

    await userEvent.type(
      await screen.findByLabelText("What needs doing?"),
      "Book the venue",
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Save for later" }),
    );

    await vi.waitFor(() => expect(held.answer).not.toBeNull());
    const rows = screen.getAllByTestId("list-row");
    expect(rows.map((row) => row.dataset.title)).toEqual(["Book the venue"]);

    held.answer?.();
    await vi.waitFor(() =>
      expect(
        screen.getByTestId("list-row").querySelector("#task-row-task-9"),
      ).not.toBeNull(),
    );
  });

  it("points to the Team page when Otto is the only one to pick", async () => {
    renderDialog();

    await userEvent.click(
      await screen.findByRole("button", { name: "Owner: Otto. Change owner" }),
    );

    const add = await screen.findByRole("menuitem", { name: /^Add an expert/ });
    expect(add.getAttribute("href")).toBe("/team");
    expect(
      within(add).getByText("Experts on your team can take tasks too."),
    ).toBeDefined();
  });

  it("forgets the picked owner when the dialog closes", async () => {
    render(<DialogHost experts={[makeExpert("maria", "Maria")]} />);

    await userEvent.click(screen.getByRole("button", { name: "Open" }));
    await userEvent.click(
      await screen.findByRole("button", { name: "Owner: Otto. Change owner" }),
    );
    await userEvent.click(
      await screen.findByRole("menuitemradio", { name: "Maria" }),
    );
    expect(
      await screen.findByRole("button", { name: "Owner: Maria. Change owner" }),
    ).toBeDefined();

    await userEvent.keyboard("{Escape}");
    await vi.waitFor(() =>
      expect(screen.queryByRole("dialog", { name: "New task" })).toBe(null),
    );

    await userEvent.click(screen.getByRole("button", { name: "Open" }));

    expect(
      await screen.findByRole("button", { name: "Owner: Otto. Change owner" }),
    ).toBeDefined();
  });
});

function BoardHarness() {
  const tasks = useListTasks(
    { limit: 100 },
    { query: { select: (res) => okData(res)?.tasks ?? [] } },
  );
  return (
    <TaskList
      tasks={tasks.data ?? []}
      experts={[]}
      selectedID={null}
      onOpen={vi.fn()}
    />
  );
}

function DialogHost({ experts }: { experts: Expert[] }) {
  const [isOpen, setIsOpen] = useState(false);
  return (
    <>
      <button type="button" onClick={() => setIsOpen(true)}>
        Open
      </button>
      <NewTaskDialog
        isOpen={isOpen}
        onOpenChange={setIsOpen}
        experts={experts}
      />
    </>
  );
}

function makeExpert(id: string, name: string): Expert {
  return {
    id,
    name,
    avatar_url: null,
    role: "Role",
    tagline: null,
    bio: null,
    skills: [],
    identity: "identity",
    voice_preferences: "",
    boundaries: "",
    protected_soul_rules: [],
    is_template: false,
    source_template_id: null,
    is_archived: false,
    workflows: [],
    pod_id: null,
  };
}
