import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import { getListExpertsMockHandler200 } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { Task } from "@/app/api/__generated__/models/task";
import type { TaskRepeat } from "@/app/api/__generated__/models/taskRepeat";
import { server } from "@/mocks/mock-server";
import { act, render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { startedTaskOf } from "../helpers";
import { useWorkAreaStore } from "../../../workArea";
import { TaskStatusLine } from "../TaskStatusLine";

const NOW = new Date(2026, 10, 17, 10, 0, 0).getTime();
const TASK_URL = "/api/proxy/api/tasks/task-1";
const HOOK_URL =
  "https://platform.example/api/integrations/generic_webhook/webhooks/wh-9/ingress";
const PR_WATCH = {
  webhook_id: "wh-1",
  provider: "github",
  trigger_block_id: "6c60ec01-8128-419e-988f-96a063ee2fea",
  trigger_config: { repo: "acme/web", events: { opened: true } },
  line: "PR opened on acme/web",
  url: null,
};
const WATCHING: TaskRepeat = {
  crons: [],
  timezone: "UTC",
  label: "PR opened on acme/web",
  paused: false,
  watch: PR_WATCH,
};

let task: Task;
let calls: string[];

function resultFor(repeat: TaskRepeat) {
  return {
    type: "task_started",
    in_background: true,
    task: {
      id: "task-1",
      title: "Review new PRs",
      status: "TODO",
      owner_expert_id: null,
      start_at: null,
      missed_start: false,
      repeat,
    },
  };
}

function serveTask() {
  server.use(
    getListExpertsMockHandler200([]),
    http.get(TASK_URL, () => HttpResponse.json(task)),
    http.post(`${TASK_URL}/repeat/pause`, () => {
      calls.push("pause");
      task = { ...task, repeat: { ...WATCHING, paused: true } };
      return HttpResponse.json(task);
    }),
    http.post(`${TASK_URL}/repeat/resume`, () => {
      calls.push("resume");
      task = { ...task, repeat: WATCHING };
      return HttpResponse.json(task);
    }),
    http.delete(`${TASK_URL}/repeat`, () => {
      calls.push("stop-watching");
      task = { ...task, status: "CANCELLED", repeat: null };
      return HttpResponse.json(task);
    }),
  );
}

function renderLine(repeat = WATCHING) {
  const started = startedTaskOf(resultFor(repeat));
  if (!started) throw new Error("The result should read as a started task");
  return render(<TaskStatusLine task={started} />);
}

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  vi.setSystemTime(NOW);
  calls = [];
  task = {
    ...makeTask("task-1", "Review new PRs", "TODO"),
    start_at: null,
    repeat: WATCHING,
  };
  serveTask();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("Watching task card in chat", () => {
  it("reads what it watches from the start_task result", () => {
    expect(startedTaskOf(resultFor(WATCHING))?.repeat).toEqual({
      label: "PR opened on acme/web",
      paused: false,
      watch: { line: "PR opened on acme/web", url: null },
    });
  });

  it("says it is watching, with no Run now and no Pause in the card", async () => {
    renderLine();

    expect(await screen.findByTestId("watching-task-line")).toBeDefined();
    expect(
      await screen.findByText("Watching · PR opened on acme/web"),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: "Run now" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Pause" })).toBeNull();
  });

  it("opens the task beside the chat when the card is clicked", async () => {
    useWorkAreaStore.getState().reset();
    renderLine();

    await userEvent.click(
      await screen.findByRole("button", { name: /^Review new PRs/ }),
    );

    const area = useWorkAreaStore.getState();
    expect(area.activeKey).toContain("task-1");
    expect(area.isFolded).toBe(false);
  });

  it("reads Paused once the task is paused", async () => {
    task = { ...task, repeat: { ...WATCHING, paused: true } };
    renderLine({ ...WATCHING, paused: true });

    expect(
      await screen.findByText("PR opened on acme/web · Paused"),
    ).toBeDefined();
  });

  it("stops watching from its menu", async () => {
    renderLine();

    await userEvent.click(
      await screen.findByRole("button", { name: "More for Review new PRs" }),
    );
    expect(
      screen.queryByRole("menuitem", { name: "Stop repeating" }),
    ).toBeNull();
    await userEvent.click(
      await screen.findByRole("menuitem", { name: "Stop watching" }),
    );

    await waitFor(() => expect(calls).toEqual(["stop-watching"]));
  });

  it("keeps the webhook URL out of the card", async () => {
    const generic = {
      ...WATCHING,
      label: "Calls to its webhook URL",
      watch: {
        ...PR_WATCH,
        provider: "generic_webhook",
        line: "Calls to its webhook URL",
        url: HOOK_URL,
      },
    };
    task = { ...task, repeat: generic };
    renderLine(generic);

    expect(
      await screen.findByText("Watching · Calls to its webhook URL"),
    ).toBeDefined();
    expect(screen.queryByText(HOOK_URL)).toBeNull();
    expect(
      screen.queryByRole("button", { name: "Copy webhook URL" }),
    ).toBeNull();
  });

  it("shows the run an event started, then goes back to watching", async () => {
    task = { ...task, status: "IN_PROGRESS" };
    renderLine();

    await waitFor(() =>
      expect(screen.getByTestId("task-status-badge").dataset.state).toBe(
        "working",
      ),
    );

    // The run ends: the task waits again for the next event.
    task = { ...task, status: "TODO" };
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000);
    });

    expect(await screen.findByTestId("watching-task-line")).toBeDefined();
    expect(screen.getByText("Watching · PR opened on acme/web")).toBeDefined();
  });
});
