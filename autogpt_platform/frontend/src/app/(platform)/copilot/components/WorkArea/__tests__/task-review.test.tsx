import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import { getListExpertsResponseMock200 } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { Task } from "@/app/api/__generated__/models/task";
import type { TaskStatus } from "@/app/api/__generated__/models/taskStatus";
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
let reviews: unknown[];
let patches: unknown[];
let refusal: { status: number; detail: string } | null;

function setTask(status: TaskStatus, extra: Partial<Task> = {}) {
  task = {
    ...makeTask("task-1", "Find design partners", status),
    result: "Found 25 partners.",
    ...extra,
  };
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
        is_unverified: task.review?.state !== "accepted",
      }),
    ),
    http.post(`${TASK_URL}/review`, async ({ request }) => {
      const body = (await request.json()) as {
        decision: string;
        note?: string;
      };
      reviews.push(body);
      if (refusal) {
        return HttpResponse.json(
          { detail: refusal.detail },
          { status: refusal.status },
        );
      }
      task =
        body.decision === "accept"
          ? { ...task, review: { state: "accepted", at: new Date() } }
          : {
              ...task,
              status: "IN_PROGRESS",
              result: null,
              review: {
                state: "changes_requested",
                note: body.note,
                at: new Date(),
              },
            };
      return HttpResponse.json(task);
    }),
    http.patch(TASK_URL, async ({ request }) => {
      const body = (await request.json()) as Partial<Task> & {
        assign_to_otto?: boolean;
      };
      patches.push(body);
      if (refusal) {
        return HttpResponse.json(
          { detail: refusal.detail },
          { status: refusal.status },
        );
      }
      task = {
        ...task,
        owner_expert_id: body.assign_to_otto
          ? null
          : (body.owner_expert_id ?? task.owner_expert_id),
      };
      return HttpResponse.json(task);
    }),
  );
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

function result() {
  return screen.findByRole("region", { name: "Result" });
}

beforeEach(() => {
  reviews = [];
  patches = [];
  refusal = null;
  mockToast.mockClear();
  serve();
});

describe("Reviewing a finished task's result", () => {
  it("accepts it: the buttons go and a quiet line says so", async () => {
    setTask("DONE");
    renderTab();
    const block = await result();
    expect(within(block).getByText("Not verified yet")).toBeDefined();

    await userEvent.click(
      within(block).getByRole("button", { name: "Accept" }),
    );

    expect(await within(block).findByText(/You accepted this/)).toBeDefined();
    expect(within(block).queryByRole("button", { name: "Accept" })).toBeNull();
    expect(within(block).queryByText("Not verified yet")).toBeNull();
    expect(reviews).toEqual([{ decision: "accept" }]);
  });

  it("asks for changes with a note and shows the task working again", async () => {
    setTask("DONE");
    renderTab();
    const block = await result();

    await userEvent.click(
      within(block).getByRole("button", { name: "Request changes" }),
    );
    const field = within(block).getByRole("textbox", {
      name: "What should change?",
    });
    await userEvent.type(field, "Only B2B partners{Enter}");

    const details = screen.getByRole("region", { name: "Details" });
    expect(await within(details).findByText("Working")).toBeDefined();
    expect(reviews).toEqual([
      { decision: "request_changes", note: "Only B2B partners" },
    ]);
  });

  it("refuses an empty note without sending it, and Escape puts it away", async () => {
    setTask("DONE");
    renderTab();
    const block = await result();

    await userEvent.click(
      within(block).getByRole("button", { name: "Request changes" }),
    );
    await userEvent.type(
      within(block).getByRole("textbox", { name: "What should change?" }),
      "{Enter}",
    );

    expect(within(block).getByText("Say what to change.")).toBeDefined();
    expect(reviews).toEqual([]);

    await userEvent.keyboard("{Escape}");
    expect(within(block).queryByRole("textbox")).toBeNull();
    expect(
      within(block).getByRole("button", { name: "Request changes" }),
    ).toBeDefined();
  });

  it("puts the result back and shows the server's reason when it says no", async () => {
    setTask("DONE");
    refusal = {
      status: 409,
      detail: "Only a finished task's result can be accepted.",
    };
    renderTab();
    const block = await result();

    await userEvent.click(
      within(block).getByRole("button", { name: "Accept" }),
    );

    await waitFor(() =>
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Couldn't accept the result",
          description: "Only a finished task's result can be accepted.",
        }),
      ),
    );
    expect(
      await within(block).findByRole("button", { name: "Accept" }),
    ).toBeDefined();
    expect(within(block).getByText("Not verified yet")).toBeDefined();
  });

  it("offers no review on a failed task, which has Try again instead", async () => {
    setTask("FAILED", { failure_reason: "No web search was available." });
    renderTab();

    expect(
      await screen.findByRole("region", { name: "What went wrong" }),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: "Accept" })).toBeNull();
    expect(
      screen.queryByRole("button", { name: "Request changes" }),
    ).toBeNull();
    expect(screen.getByRole("button", { name: "Try again" })).toBeDefined();
  });
});

describe("Changing who has the task", () => {
  it("gives a to-do task to an expert", async () => {
    setTask("TODO");
    renderTab();

    await userEvent.click(
      await screen.findByRole("button", { name: "Owner: Otto. Change owner" }),
    );
    await userEvent.click(
      await screen.findByRole("menuitemradio", { name: "Leo" }),
    );

    expect(
      await screen.findByRole("button", { name: "Owner: Leo. Change owner" }),
    ).toBeDefined();
    expect(patches).toEqual([{ owner_expert_id: "expert-leo" }]);
  });

  it("puts the owner back with the server's reason when it refuses", async () => {
    setTask("TODO");
    refusal = {
      status: 409,
      detail:
        "It is being worked on. Stop it before giving it to someone else.",
    };
    renderTab();

    await userEvent.click(
      await screen.findByRole("button", { name: "Owner: Otto. Change owner" }),
    );
    await userEvent.click(
      await screen.findByRole("menuitemradio", { name: "Leo" }),
    );

    await waitFor(() =>
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Couldn't change who has it",
          description:
            "It is being worked on. Stop it before giving it to someone else.",
        }),
      ),
    );
    expect(
      await screen.findByRole("button", { name: "Owner: Otto. Change owner" }),
    ).toBeDefined();
  });

  it("keeps the owner read-only while the task is worked on", async () => {
    setTask("IN_PROGRESS", { result: null });
    renderTab();

    const details = await screen.findByRole("region", { name: "Details" });
    expect(
      await within(details).findByText("Stop it to change who has it"),
    ).toBeDefined();
    expect(
      within(details).queryByRole("button", { name: /Change owner/ }),
    ).toBeNull();
  });
});
