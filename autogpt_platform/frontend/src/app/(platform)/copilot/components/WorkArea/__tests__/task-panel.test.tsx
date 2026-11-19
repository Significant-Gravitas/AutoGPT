import { QueryClient } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import { getListExpertsResponseMock200 } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getListTasksQueryKey } from "@/app/api/__generated__/endpoints/tasks/tasks";
import type { Task } from "@/app/api/__generated__/models/task";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { expectTypingKept } from "@/tests/integrations/typing";
import { taskTabKey, useWorkAreaStore } from "../../../workArea";
import { TaskTab } from "../TaskTab";

const TASK_URL = "/api/proxy/api/tasks/task-1";
const [leoMock] = getListExpertsResponseMock200();
const LEO = { ...leoMock, id: "expert-leo", name: "Leo" };

let task: Task;
let patches: unknown[];
let archived: boolean;
let release: () => void;

function serve({ hold = false }: { hold?: boolean } = {}) {
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  if (!hold) release();
  server.use(
    http.get(TASK_URL, async () => {
      await held;
      return HttpResponse.json(task);
    }),
    http.get(`${TASK_URL}/record`, async () => {
      await held;
      return HttpResponse.json({
        task,
        activity: [],
        artifacts: [],
        evidence: [],
      });
    }),
    http.patch(TASK_URL, async ({ request }) => {
      const body = (await request.json()) as Partial<Task>;
      patches.push(body);
      task = { ...task, ...body };
      return HttpResponse.json(task);
    }),
    http.delete(TASK_URL, () => {
      archived = true;
      return HttpResponse.json(task);
    }),
  );
}

function renderTab(queryClient?: QueryClient) {
  return render(
    <TaskTab
      taskID="task-1"
      sessionID={null}
      experts={[LEO]}
      onOpenArtifact={vi.fn()}
    />,
    { queryClient },
  );
}

function about() {
  return screen.findByRole("region", { name: "About" });
}

beforeEach(() => {
  patches = [];
  archived = false;
  task = {
    ...makeTask("task-1", "Plan the launch", "IN_PROGRESS", LEO.id),
    description: "Keep it friendly.",
  };
  useWorkAreaStore.getState().reset();
  useWorkAreaStore.getState().openTask("task-1", "Plan the launch");
});

afterEach(() => release());

describe("Task tab: what the task is for", () => {
  it("shows the description and edits it in place; Enter saves", async () => {
    serve();
    renderTab();
    const block = await about();

    await userEvent.click(
      within(block).getByRole("button", { name: /^Description/ }),
    );
    const field = within(block).getByRole("textbox", { name: "Description" });
    expect((field as HTMLTextAreaElement).value).toBe("Keep it friendly.");
    await expectTypingKept(field as HTMLTextAreaElement, " And short 日本");

    await userEvent.keyboard("{Enter}");

    expect(
      await within(block).findByText("Keep it friendly. And short 日本"),
    ).toBeDefined();
    expect(within(block).queryByRole("textbox")).toBeNull();
    expect(patches).toEqual([
      { description: "Keep it friendly. And short 日本" },
    ]);
  });

  it("keeps the old words on Escape", async () => {
    serve();
    renderTab();
    const block = await about();

    await userEvent.click(within(block).getByRole("button", { name: /^Goal/ }));
    await userEvent.type(
      within(block).getByRole("textbox", { name: "Goal" }),
      "Ship by Friday",
    );
    await userEvent.keyboard("{Escape}");

    expect(within(block).getByText("Add a goal")).toBeDefined();
    expect(within(block).queryByText("Ship by Friday")).toBeNull();
    expect(patches).toEqual([]);
  });

  it("saves when the field is left", async () => {
    serve();
    renderTab();
    const block = await about();

    await userEvent.click(
      within(block).getByRole("button", { name: /^Done when/ }),
    );
    await userEvent.type(
      within(block).getByRole("textbox", { name: "Done when" }),
      "The page is live",
    );
    await userEvent.tab();

    expect(await within(block).findByText("The page is live")).toBeDefined();
    expect(patches).toEqual([{ done_when: "The page is live" }]);
  });
});

describe("Task tab header", () => {
  it("has the card's menu: stop, rename and archive", async () => {
    serve();
    renderTab();

    await userEvent.click(
      await screen.findByRole("button", {
        name: "Actions for Plan the launch",
      }),
    );
    expect(await screen.findByRole("menuitem", { name: "Stop" })).toBeDefined();
    await userEvent.click(screen.getByRole("menuitem", { name: "Rename" }));
    const title = await screen.findByRole("textbox", { name: "Task title" });
    await userEvent.clear(title);
    await userEvent.type(title, "Plan the big launch{Enter}");

    expect(
      await screen.findByRole("heading", { name: "Plan the big launch" }),
    ).toBeDefined();
    expect(patches).toEqual([{ title: "Plan the big launch" }]);

    await userEvent.click(
      screen.getByRole("button", { name: "Actions for Plan the big launch" }),
    );
    await userEvent.click(
      await screen.findByRole("menuitem", { name: "Archive" }),
    );
    await userEvent.click(
      await screen.findByRole("button", { name: "Archive task" }),
    );

    await waitFor(() => expect(archived).toBe(true));
    await waitFor(() =>
      expect(
        useWorkAreaStore
          .getState()
          .tabs.some((tab) => tab.key === taskTabKey("task-1")),
      ).toBe(false),
    );
  });
});

describe("Task tab before its record arrives", () => {
  it("draws the task a list already read, at once", async () => {
    serve({ hold: true });
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    queryClient.setQueryData(getListTasksQueryKey({ limit: 5 }), {
      status: 200,
      data: { tasks: [task] },
      headers: new Headers(),
    });
    renderTab(queryClient);

    expect(
      await screen.findByRole("heading", { name: "Plan the launch" }),
    ).toBeDefined();
    const details = screen.getByRole("region", { name: "Details" });
    expect(within(details).getByText("Working")).toBeDefined();
    expect(within(details).getByText("Leo")).toBeDefined();
    expect(screen.getByText("Keep it friendly.")).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Actions for Plan the launch" }),
    ).toBeDefined();
  });

  it("shows the name the tab was opened with, the rest as placeholders", async () => {
    serve({ hold: true });
    renderTab();

    expect(
      await screen.findByRole("heading", { name: "Plan the launch" }),
    ).toBeDefined();
    expect(screen.queryByRole("region", { name: "Details" })).toBeNull();

    release();
    expect(
      await screen.findByRole("region", { name: "Details" }),
    ).toBeDefined();
  });
});
