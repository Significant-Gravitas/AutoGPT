import { http, HttpResponse } from "msw";
import type { ReactElement } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import type { Task } from "@/app/api/__generated__/models/task";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";

import SchedulesPage from "../page";

const { toast, push } = vi.hoisted(() => ({ toast: vi.fn(), push: vi.fn() }));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast,
  useToast: () => ({ toast }),
}));
vi.mock("next/navigation", async (importOriginal) => ({
  ...(await importOriginal<typeof import("next/navigation")>()),
  useRouter: () => ({ push, replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/schedules",
  useSearchParams: () => new URLSearchParams(),
}));

const API = "/api/proxy/api";
const HOUR = 3_600_000;
const NOW = Date.now();
const ROW = { selector: "button[data-id]" };

function task(overrides: Partial<Task>): Task {
  return {
    id: "task-1",
    user_id: "user-1",
    title: "Weekly report",
    status: "TODO",
    created_by: "USER",
    created_at: new Date(NOW - 48 * HOUR),
    updated_at: new Date(NOW - HOUR),
    session_ids: ["chat-1"],
    work_session_id: "chat-1",
    ...overrides,
  };
}

const REPEATING = task({
  start_at: new Date(NOW + 2 * HOUR),
  repeat: { crons: ["0 9 * * 1"], timezone: "UTC", label: "Every Monday" },
});
const ONCE = task({
  id: "task-2",
  title: "Send the board pack",
  start_at: new Date(NOW + 5 * HOUR),
});
const WORKFLOW: GraphExecutionJobInfo = {
  id: "sched-1",
  name: "Morning digest",
  user_id: "user-1",
  graph_id: "graph-1",
  graph_version: 3,
  agent_name: "Digest",
  cron: "0 8 * * *",
  input_data: { topic: "news" },
  next_run_time: new Date(NOW + 3 * HOUR).toISOString(),
  timezone: "Europe/London",
};

/** The lists the page reads; each answers with what the test holds now. */
function serveLists(lists: {
  tasks: Task[];
  workflows: GraphExecutionJobInfo[];
}) {
  server.use(
    http.get(`${API}/tasks`, () => HttpResponse.json({ tasks: lists.tasks })),
    http.get(`${API}/schedules`, () => HttpResponse.json(lists.workflows)),
    http.get(`${API}/schedules/upcoming`, () =>
      HttpResponse.json({ schedules: [] }),
    ),
    http.get(`${API}/tasks/:id/record`, () =>
      HttpResponse.json({ task: REPEATING, runs: [] }),
    ),
    http.get(`${API}/graphs/:id/executions`, () =>
      HttpResponse.json({
        executions: [],
        pagination: {
          total_items: 0,
          total_pages: 0,
          current_page: 1,
          page_size: 10,
        },
      }),
    ),
  );
}

/** A request that waits until the test lets it answer. */
function held() {
  let release: () => void = () => {};
  const gate = new Promise<void>((resolve) => (release = resolve));
  return { gate, release };
}

function lastToast() {
  return toast.mock.calls.at(-1)?.[0] as {
    title: string;
    variant?: string;
    action?: ReactElement<{ href?: string; onClick?: () => void }>;
  };
}

function switchOf(name: string) {
  return screen.getByLabelText(new RegExp(`^${name}: (Pause|Resume)$`), {
    selector: "button",
  });
}

beforeEach(() => {
  toast.mockClear();
  push.mockClear();
});

describe("Schedules page actions", () => {
  it("pauses at once, then says so", async () => {
    serveLists({ tasks: [REPEATING], workflows: [] });
    const { gate, release } = held();
    server.use(
      http.post(`${API}/tasks/task-1/repeat/pause`, async () => {
        await gate;
        return HttpResponse.json(REPEATING);
      }),
    );
    render(<SchedulesPage />);

    fireEvent.click(
      await screen.findByLabelText("Weekly report: Pause", {
        selector: "button",
      }),
    );
    // Off before the server has answered.
    await waitFor(() =>
      expect(switchOf("Weekly report").getAttribute("aria-checked")).toBe(
        "false",
      ),
    );
    expect(toast).not.toHaveBeenCalled();

    release();
    await waitFor(() =>
      expect(lastToast()).toMatchObject({
        title: "Paused Weekly report",
        variant: "success",
      }),
    );
  });

  it("puts the switch back and says why when pausing fails", async () => {
    serveLists({ tasks: [REPEATING], workflows: [] });
    server.use(
      http.post(`${API}/tasks/task-1/repeat/pause`, () =>
        HttpResponse.json({ detail: "Not allowed" }, { status: 409 }),
      ),
    );
    render(<SchedulesPage />);

    fireEvent.click(
      await screen.findByLabelText("Weekly report: Pause", {
        selector: "button",
      }),
    );

    await waitFor(() =>
      expect(lastToast()).toMatchObject({
        title: "The task could not be paused",
        variant: "destructive",
      }),
    );
    await waitFor(() =>
      expect(switchOf("Weekly report").getAttribute("aria-checked")).toBe(
        "true",
      ),
    );
  });

  it("stops a repeating task after a second click, at once", async () => {
    const lists = { tasks: [REPEATING, ONCE], workflows: [] };
    serveLists(lists);
    const stopped = vi.fn();
    const { gate, release } = held();
    server.use(
      http.delete(`${API}/tasks/task-1/repeat`, async () => {
        stopped();
        await gate;
        lists.tasks = [ONCE];
        return HttpResponse.json({ ...REPEATING, repeat: null });
      }),
    );
    render(<SchedulesPage />);

    fireEvent.click(
      await screen.findByLabelText(/^Weekly report, Every Monday/, ROW),
    );
    fireEvent.click(await screen.findByText("Stop repeating"));
    expect(stopped).not.toHaveBeenCalled();
    expect(screen.getByText("Open task")).toBeDefined();
    fireEvent.click(screen.getByText("Stop repeating for good"));

    // Gone while the server is still answering.
    await waitFor(() => expect(stopped).toHaveBeenCalledTimes(1));
    await waitFor(() =>
      expect(screen.queryByLabelText(/^Weekly report, /, ROW)).toBeNull(),
    );
    release();
    await waitFor(() =>
      expect(lastToast().title).toBe("Weekly report stopped repeating"),
    );
  });

  it("cancels a task waiting for its one start", async () => {
    const lists = { tasks: [REPEATING, ONCE], workflows: [] };
    serveLists(lists);
    const { gate, release } = held();
    server.use(
      http.post(`${API}/tasks/task-2/cancel`, async () => {
        await gate;
        lists.tasks = [REPEATING];
        return HttpResponse.json({ ...ONCE, status: "CANCELLED" });
      }),
    );
    render(<SchedulesPage />);

    fireEvent.click(
      await screen.findByLabelText(/^Send the board pack, Once/, ROW),
    );
    fireEvent.click(await screen.findByText("Cancel"));
    fireEvent.click(screen.getByText("Cancel this task"));

    await waitFor(() =>
      expect(screen.queryByLabelText(/^Send the board pack, /, ROW)).toBeNull(),
    );
    release();
    await waitFor(() =>
      expect(lastToast().title).toBe("Send the board pack was cancelled"),
    );
  });

  it("deletes a workflow schedule at once and puts it back with Undo", async () => {
    const lists = { tasks: [] as Task[], workflows: [WORKFLOW] };
    serveLists(lists);
    const created = vi.fn();
    const { gate, release } = held();
    server.use(
      http.delete(`${API}/schedules/sched-1`, async () => {
        await gate;
        lists.workflows = [];
        return HttpResponse.json({});
      }),
      http.post(`${API}/graphs/graph-1/schedules`, async ({ request }) => {
        created(await request.json());
        return HttpResponse.json({ ...WORKFLOW, id: "sched-2" });
      }),
    );
    render(<SchedulesPage />);

    fireEvent.click(await screen.findByLabelText(/^Morning digest, /, ROW));
    fireEvent.click(await screen.findByText("Delete"));
    fireEvent.click(screen.getByText("Delete this schedule"));

    await waitFor(() =>
      expect(screen.queryByLabelText(/^Morning digest, /, ROW)).toBeNull(),
    );
    expect(toast).not.toHaveBeenCalled();
    release();
    await waitFor(() =>
      expect(lastToast().title).toBe("Morning digest was deleted"),
    );
    lastToast().action?.props.onClick?.();

    await waitFor(() =>
      expect(created).toHaveBeenCalledWith({
        graph_version: 3,
        name: "Morning digest",
        cron: "0 8 * * *",
        inputs: { topic: "news" },
        credentials: {},
        timezone: "Europe/London",
      }),
    );
  });

  it("puts a deleted workflow schedule back when the delete fails", async () => {
    serveLists({ tasks: [], workflows: [WORKFLOW] });
    server.use(
      http.delete(`${API}/schedules/sched-1`, () =>
        HttpResponse.json({ detail: "Server error" }, { status: 500 }),
      ),
    );
    render(<SchedulesPage />);

    fireEvent.click(await screen.findByLabelText(/^Morning digest, /, ROW));
    fireEvent.click(await screen.findByText("Delete"));
    fireEvent.click(screen.getByText("Delete this schedule"));

    await waitFor(() =>
      expect(lastToast()).toMatchObject({
        title: "The schedule could not be deleted",
        variant: "destructive",
      }),
    );
    expect(
      await screen.findByLabelText(/^Morning digest, /, ROW),
    ).toBeDefined();
  });

  it("says a workflow started and links to its run", async () => {
    serveLists({ tasks: [], workflows: [WORKFLOW] });
    server.use(
      http.post(`${API}/graphs/graph-1/execute/3`, () =>
        HttpResponse.json({ id: "exec-9", graph_id: "graph-1" }),
      ),
    );
    render(<SchedulesPage />);

    fireEvent.click(await screen.findByLabelText(/^Morning digest, /, ROW));
    fireEvent.click(await screen.findByText("Run now"));

    await waitFor(() =>
      expect(lastToast().title).toBe("Started Morning digest"),
    );
    expect(lastToast().action?.props.href).toBe("/runs/graph-1/exec-9");
  });

  it("says a task started and links to it", async () => {
    serveLists({ tasks: [REPEATING], workflows: [] });
    server.use(
      http.post(`${API}/tasks/task-1/run`, () =>
        HttpResponse.json({ ...REPEATING, status: "IN_PROGRESS" }),
      ),
    );
    render(<SchedulesPage />);

    fireEvent.click(
      await screen.findByLabelText(/^Weekly report, Every Monday/, ROW),
    );
    fireEvent.click(await screen.findByText("Run now"));

    await waitFor(() =>
      expect(lastToast().title).toBe("Started Weekly report"),
    );
    expect(lastToast().action?.props.href).toBe(
      "/copilot?sessionId=chat-1&task=task-1",
    );
  });

  it("offers an example to start from when there are no schedules", async () => {
    serveLists({ tasks: [], workflows: [] });
    render(<SchedulesPage />);

    expect(
      await screen.findByText(
        /For example: “Every Monday at 9am, send me a summary of last week\.”/,
      ),
    ).toBeDefined();
    fireEvent.click(screen.getByText("Ask Otto this"));

    expect(push).toHaveBeenCalledWith(
      "/home#prompt=Every%20Monday%20at%209am%2C%20send%20me%20a%20summary%20of%20last%20week",
    );
  });
});
