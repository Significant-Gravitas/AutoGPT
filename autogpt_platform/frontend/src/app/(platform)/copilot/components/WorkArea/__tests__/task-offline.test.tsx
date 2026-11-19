import { onlineManager } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import { getListExpertsResponseMock200 } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { Task } from "@/app/api/__generated__/models/task";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { TaskTab } from "../TaskTab";

const mockToast = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/components/molecules/Toast/use-toast")
  >()),
  toast: mockToast,
}));

const TASK_URL = "/api/proxy/api/tasks/task-1";
const [leoMock] = getListExpertsResponseMock200();
const LEO = { ...leoMock, id: "expert-leo", name: "Leo", is_archived: false };

let task: Task;
let sent: string[];
let offline: boolean;

function reply(kind: string, body: () => Response) {
  sent.push(kind);
  return offline ? HttpResponse.error() : body();
}

function serve() {
  server.use(
    http.get(TASK_URL, () => HttpResponse.json(task)),
    http.get(`${TASK_URL}/record`, () =>
      HttpResponse.json({
        task,
        activity: [],
        artifacts: [],
        evidence: [],
        is_unverified: true,
      }),
    ),
    http.patch(TASK_URL, () => reply("patch", () => HttpResponse.json(task))),
    http.post(`${TASK_URL}/review`, () =>
      reply("review", () => HttpResponse.json(task)),
    ),
  );
}

function goOffline() {
  offline = true;
  act(() => onlineManager.setOnline(false));
}

function goOnline() {
  offline = false;
  act(() => onlineManager.setOnline(true));
}

function unreachableToast() {
  const call = mockToast.mock.calls.find(
    ([props]) => props.title === "Couldn't reach AutoGPT",
  );
  return call?.[0];
}

function renderTab() {
  return render(
    <TaskTab
      taskID="task-1"
      sessionID={null}
      experts={[LEO]}
      onOpenArtifact={vi.fn()}
    />,
  );
}

beforeEach(() => {
  sent = [];
  offline = false;
  mockToast.mockClear();
  serve();
});

afterEach(() => goOnline());

describe("The task panel offline", () => {
  it("puts a saved field back at once and sends nothing later", async () => {
    task = { ...makeTask("task-1", "Plan the launch", "TODO"), goal: null };
    renderTab();
    const about = await screen.findByRole("region", { name: "About" });
    goOffline();

    await userEvent.click(within(about).getByRole("button", { name: /Goal/ }));
    await userEvent.type(
      within(about).getByRole("textbox", { name: "Goal" }),
      "Ship by Friday{Enter}",
    );

    await waitFor(() => expect(unreachableToast()).toBeDefined());
    expect(await within(about).findByText("Add a goal")).toBeDefined();
    expect(unreachableToast()?.action).toBeDefined();

    goOnline();
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(sent).toEqual(["patch"]);
  });

  it("puts an accept back and leaves the buttons", async () => {
    task = {
      ...makeTask("task-1", "Plan the launch", "DONE"),
      result: "Found 25 partners.",
    };
    renderTab();
    const result = await screen.findByRole("region", { name: "Result" });
    goOffline();

    await userEvent.click(
      within(result).getByRole("button", { name: "Accept" }),
    );

    await waitFor(() => expect(unreachableToast()).toBeDefined());
    expect(
      await within(result).findByRole("button", { name: "Accept" }),
    ).toBeDefined();
    expect(within(result).getByText("Not verified yet")).toBeDefined();
    goOnline();
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(sent).toEqual(["review"]);
  });

  it("puts the owner back", async () => {
    task = makeTask("task-1", "Plan the launch", "TODO");
    renderTab();
    goOffline();

    await userEvent.click(
      await screen.findByRole("button", { name: "Owner: Otto. Change owner" }),
    );
    await userEvent.click(
      await screen.findByRole("menuitemradio", { name: "Leo" }),
    );

    await waitFor(() => expect(unreachableToast()).toBeDefined());
    expect(
      await screen.findByRole("button", { name: "Owner: Otto. Change owner" }),
    ).toBeDefined();
    goOnline();
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(sent).toEqual(["patch"]);
  });
});
