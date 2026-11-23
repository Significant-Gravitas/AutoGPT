import { focusManager, QueryClient } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import type { Task } from "@/app/api/__generated__/models/task";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { taskTabKey, useWorkAreaStore } from "../../../workArea";
import { TaskTab } from "../TaskTab";

const mockToast = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/components/molecules/Toast/use-toast")
  >()),
  toast: mockToast,
}));

const TASK_URL = "/api/proxy/api/tasks/task-1";
const RAW_ID = "6f57ed04-1b2c-4d5e-8f90-a1b2c3d4e5f6";
const GONE = { detail: `Task ${RAW_ID} not found` };

let task: Task;
let recordReads: number;
let gone: boolean;

function serve() {
  server.use(
    http.get(TASK_URL, () =>
      gone ? HttpResponse.json(GONE, { status: 404 }) : HttpResponse.json(task),
    ),
    http.get(`${TASK_URL}/record`, () => {
      recordReads += 1;
      return gone
        ? HttpResponse.json(GONE, { status: 404 })
        : HttpResponse.json({
            task,
            activity: [],
            artifacts: [],
            evidence: [],
          });
    }),
    http.patch(TASK_URL, () => HttpResponse.json(GONE, { status: 404 })),
  );
}

function isOpen() {
  return useWorkAreaStore
    .getState()
    .tabs.some((tab) => tab.key === taskTabKey("task-1"));
}

function toastTexts() {
  return JSON.stringify(mockToast.mock.calls);
}

function renderTab(queryClient?: QueryClient) {
  return render(
    <TaskTab
      taskID="task-1"
      sessionID={null}
      experts={[]}
      onOpenArtifact={vi.fn()}
    />,
    { queryClient },
  );
}

beforeEach(() => {
  task = { ...makeTask("task-1", "Plan the launch", "DONE"), goal: "Ship" };
  recordReads = 0;
  gone = false;
  mockToast.mockClear();
  useWorkAreaStore.getState().reset();
  useWorkAreaStore.getState().openTask("task-1", "Plan the launch");
  serve();
});

afterEach(() => focusManager.setFocused(undefined));

describe("A task archived or deleted elsewhere", () => {
  it("closes its tab and says so once when the panel reads it", async () => {
    gone = true;
    renderTab();

    await waitFor(() => expect(isOpen()).toBe(false));
    await waitFor(() =>
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({ title: "This task no longer exists" }),
      ),
    );
    const titles = mockToast.mock.calls.map(([call]) => call.title);
    expect(new Set(titles)).toEqual(new Set(["This task no longer exists"]));
    expect(toastTexts()).not.toContain(RAW_ID);
  });

  it("closes its tab when a save finds it gone, with no id shown", async () => {
    renderTab();
    const about = await screen.findByRole("region", { name: "About" });

    await userEvent.click(within(about).getByRole("button", { name: /Goal/ }));
    await userEvent.type(
      within(about).getByRole("textbox", { name: "Goal" }),
      " fast{Enter}",
    );

    await waitFor(() => expect(isOpen()).toBe(false));
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ title: "This task no longer exists" }),
    );
    expect(toastTexts()).not.toContain(RAW_ID);
  });

  it("reads the task again when the window gets focus back", async () => {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false, staleTime: 60_000 } },
    });
    renderTab(queryClient);
    await screen.findByRole("region", { name: "About" });
    const reads = recordReads;

    act(() => {
      focusManager.setFocused(false);
      focusManager.setFocused(true);
    });

    await waitFor(() => expect(recordReads).toBeGreaterThan(reads));
  });
});
