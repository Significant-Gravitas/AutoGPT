import {
  getGetBrainDumpSuggestedTasksMockHandler,
  getUpdateBrainDumpSuggestedTasksMockHandler,
} from "@/app/api/__generated__/endpoints/brain-dump/brain-dump.msw";
import type { SuggestedTask } from "@/app/api/__generated__/models/suggestedTask";
import type { SuggestedTasksUpdate } from "@/app/api/__generated__/models/suggestedTasksUpdate";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import { takeExpertThreadSource } from "@/services/experts/expert-thread-source";
import {
  act,
  renderHook,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ExpertIdentity } from "../../../useExpertMap";
import { useSuggestedTasks } from "../useSuggestedTasks";

const capture = vi.hoisted(() => vi.fn());
vi.mock("posthog-js", () => ({ default: { capture } }));

type HookArgs = Parameters<typeof useSuggestedTasks>[0];
type OnSend = (message: string) => Promise<string | undefined>;

const MARIA: ExpertIdentity = {
  id: "expert-maria",
  name: "Maria",
  avatarUrl: null,
  role: "Bookkeeper",
  isArchived: false,
  readOnlyReason: null,
};

// Removed first, so a task's rank in the stored order and its position among
// the rows the user sees differ by one.
const TASKS: SuggestedTask[] = [
  {
    id: "t0",
    title: "Tidy the shared drive",
    prompt: "Sort the shared drive into one folder per client.",
    state: "dismissed",
  },
  {
    id: "t1",
    title: "Chase unpaid invoices",
    prompt: "Chase every unpaid invoice with a polite follow-up.",
    why: "Three invoices are past due.",
    state: "open",
    expert_id: MARIA.id,
    expert_template_id: "template-bookkeeper",
    expert_name: "Maria",
  },
  {
    id: "t2",
    title: "Watch competitor pricing",
    prompt: "Check competitor pricing weekly and tell me what moved.",
    state: "open",
  },
  {
    id: "t3",
    title: "Draft the launch post",
    prompt: "Draft a launch post for the new plan.",
    state: "open",
    expert_template_id: "template-writer",
  },
];

const FAILURE_TITLE = "Could not start that task";

let stored: SuggestedTask[] = [];
let writes: SuggestedTasksUpdate[] = [];
let client: QueryClient;

beforeEach(() => {
  capture.mockReset();
  takeExpertThreadSource();
  stored = TASKS.map((task) => ({ ...task }));
  writes = [];
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  server.use(
    getGetBrainDumpSuggestedTasksMockHandler(() => ({
      tasks: stored,
      stored: true,
    })),
    getUpdateBrainDumpSuggestedTasksMockHandler(async ({ request }) => {
      const update = (await request.json()) as SuggestedTasksUpdate;
      writes.push(update);
      stored = stored.map((task) =>
        update.task_ids.includes(task.id)
          ? {
              ...task,
              state: update.state,
              session_id: update.session_id ?? task.session_id,
            }
          : task,
      );
      return { tasks: stored, stored: true };
    }),
  );
});

afterEach(() => {
  vi.useRealTimers();
});

function Wrapper({ children }: { children: ReactNode }) {
  return (
    <QueryClientProvider client={client}>
      {children}
      <Toaster />
    </QueryClientProvider>
  );
}

async function renderTasks(args: Partial<HookArgs> = {}) {
  const initialProps: HookArgs = {
    surface: "home",
    enabled: true,
    onSend: vi.fn<OnSend>(async () => "session-1"),
    experts: [MARIA],
    expertsById: new Map([[MARIA.id, MARIA]]),
    isRosterLoaded: true,
    hasRosterFailed: false,
    recipientID: null,
    selectRecipient: vi.fn(),
    canSend: true,
    ...args,
  };
  const view = renderHook((props: HookArgs) => useSuggestedTasks(props), {
    initialProps,
    wrapper: Wrapper,
  });
  await waitFor(() => expect(view.result.current.tasks).toHaveLength(3));
  return { ...view, props: initialProps };
}

function shownTask(tasks: SuggestedTask[], id: string) {
  const task = tasks.find((candidate) => candidate.id === id);
  if (!task) throw new Error(`Task ${id} is not shown`);
  return task;
}

function events(name: string) {
  return capture.mock.calls
    .filter(([event]) => event === name)
    .map(([, properties]) => properties);
}

describe("useSuggestedTasks — who takes a task", () => {
  it("hands a task to its expert once the composer has switched to them", async () => {
    const sources: (string | null)[] = [];
    const onSend = vi.fn<OnSend>(async () => {
      sources.push(takeExpertThreadSource());
      return "session-1";
    });
    const selectRecipient = vi.fn();
    const { result, rerender, props } = await renderTasks({
      onSend,
      selectRecipient,
    });

    act(() => result.current.sendTask(shownTask(result.current.tasks, "t1")));

    expect(selectRecipient).toHaveBeenCalledExactlyOnceWith(MARIA.id);
    expect(onSend).not.toHaveBeenCalled();
    expect(result.current.sendingId).toBe("t1");

    rerender({ ...props, recipientID: MARIA.id });

    await waitFor(() =>
      expect(onSend).toHaveBeenCalledExactlyOnceWith(TASKS[1].prompt),
    );
    expect(sources).toEqual(["suggested_task"]);
    await waitFor(() =>
      expect(writes).toEqual([
        { task_ids: ["t1"], state: "sent", session_id: "session-1" },
      ]),
    );
    expect(shownTask(result.current.tasks, "t1").state).toBe("sent");
    expect(result.current.sendingId).toBeNull();

    rerender({ ...props, recipientID: null });
    rerender({ ...props, recipientID: MARIA.id });

    expect(onSend).toHaveBeenCalledOnce();
    expect(selectRecipient).toHaveBeenCalledOnce();
  });

  it("starts nothing before the roster has answered", async () => {
    // Until then Maria would read as gone and her task would go to Otto.
    const onSend = vi.fn<OnSend>(async () => "session-1");
    const selectRecipient = vi.fn();
    const { result } = await renderTasks({
      onSend,
      selectRecipient,
      isRosterLoaded: false,
    });

    act(() => result.current.sendTask(shownTask(result.current.tasks, "t1")));

    expect(result.current.sendingId).toBeNull();
    expect(selectRecipient).not.toHaveBeenCalled();
    expect(onSend).not.toHaveBeenCalled();
  });

  it("waits for the page to be able to send to the new recipient", async () => {
    // Right after the switch the page is still resolving Maria, and a send
    // then would start no chat.
    const onSend = vi.fn<OnSend>(async () => "session-1");
    const { result, rerender, props } = await renderTasks({ onSend });

    act(() => result.current.sendTask(shownTask(result.current.tasks, "t1")));
    rerender({ ...props, recipientID: MARIA.id, canSend: false });
    expect(onSend).not.toHaveBeenCalled();

    rerender({ ...props, recipientID: MARIA.id, canSend: true });
    await waitFor(() => expect(onSend).toHaveBeenCalledOnce());
  });

  it("puts the recipient back when the task does not start", async () => {
    const onSend = vi.fn<OnSend>(async () => undefined);
    const selectRecipient = vi.fn();
    const { result, rerender, props } = await renderTasks({
      onSend,
      selectRecipient,
    });

    act(() => result.current.sendTask(shownTask(result.current.tasks, "t1")));
    rerender({ ...props, selectRecipient, recipientID: MARIA.id });

    await waitFor(() => expect(onSend).toHaveBeenCalledOnce());
    await waitFor(() => expect(selectRecipient).toHaveBeenLastCalledWith(null));
    expect(result.current.sendingId).toBeNull();
  });

  it("sends at once when the task's expert is already the recipient", async () => {
    const onSend = vi.fn<OnSend>(async () => "session-1");
    const selectRecipient = vi.fn();
    const { result } = await renderTasks({
      onSend,
      selectRecipient,
      recipientID: MARIA.id,
    });

    act(() => result.current.sendTask(shownTask(result.current.tasks, "t1")));

    expect(onSend).toHaveBeenCalledExactlyOnceWith(TASKS[1].prompt);
    expect(selectRecipient).not.toHaveBeenCalled();
    await waitFor(() => expect(result.current.sendingId).toBeNull());
  });

  it("switches back to Otto before sending one of Otto's tasks", async () => {
    const sources: (string | null)[] = [];
    const onSend = vi.fn<OnSend>(async () => {
      sources.push(takeExpertThreadSource());
      return "session-2";
    });
    const selectRecipient = vi.fn();
    const { result, rerender, props } = await renderTasks({
      onSend,
      selectRecipient,
      recipientID: MARIA.id,
    });

    act(() => result.current.sendTask(shownTask(result.current.tasks, "t2")));

    expect(selectRecipient).toHaveBeenCalledExactlyOnceWith(null);
    expect(onSend).not.toHaveBeenCalled();

    rerender({ ...props, recipientID: null });

    await waitFor(() =>
      expect(onSend).toHaveBeenCalledExactlyOnceWith(TASKS[2].prompt),
    );
    expect(sources).toEqual([null]);
    await waitFor(() =>
      expect(writes).toEqual([
        { task_ids: ["t2"], state: "sent", session_id: "session-2" },
      ]),
    );
  });

  it("gives up with a failure toast when the recipient never switches", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const onSend = vi.fn<OnSend>(async () => "session-1");
    const { result, rerender, props } = await renderTasks({ onSend });

    act(() => result.current.sendTask(shownTask(result.current.tasks, "t1")));

    expect(result.current.sendingId).toBe("t1");
    expect(screen.queryByText(FAILURE_TITLE)).toBeNull();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000);
    });

    expect(result.current.sendingId).toBeNull();
    expect(await screen.findByText(FAILURE_TITLE)).toBeDefined();
    expect(
      screen.getByText(
        "The task's expert could not be selected — it is still in your suggestions.",
      ),
    ).toBeDefined();

    // A switch that lands after the give-up does not send the task after all.
    rerender({ ...props, recipientID: MARIA.id });

    expect(onSend).not.toHaveBeenCalled();
    expect(shownTask(result.current.tasks, "t1").state).toBe("open");
    expect(writes).toEqual([]);
  });

  it("puts the recipient back when the switch lands but the page cannot send before giving up", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const onSend = vi.fn<OnSend>(async () => "session-1");
    const selectRecipient = vi.fn();
    const { result, rerender, props } = await renderTasks({
      onSend,
      selectRecipient,
    });

    act(() => result.current.sendTask(shownTask(result.current.tasks, "t1")));
    expect(selectRecipient.mock.calls).toEqual([[MARIA.id]]);
    // The switch lands, but the page is still resolving Maria.
    rerender({ ...props, recipientID: MARIA.id, canSend: false });

    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000);
    });

    expect(await screen.findByText(FAILURE_TITLE)).toBeDefined();
    expect(selectRecipient.mock.calls).toEqual([[MARIA.id], [null]]);
    expect(onSend).not.toHaveBeenCalled();
    expect(result.current.sendingId).toBeNull();
  });
});

describe("useSuggestedTasks — an expert who isn't on the team", () => {
  it("sends the task to Otto and says Maria isn't on the team", async () => {
    const sources: (string | null)[] = [];
    const onSend = vi.fn<OnSend>(async () => {
      sources.push(takeExpertThreadSource());
      return "session-1";
    });
    const selectRecipient = vi.fn();
    const { result } = await renderTasks({
      onSend,
      selectRecipient,
      experts: [],
    });

    act(() => result.current.sendTask(shownTask(result.current.tasks, "t1")));

    expect(
      await screen.findByText(
        "Maria isn't on your team, so Otto has this one.",
      ),
    ).toBeDefined();
    expect(onSend).toHaveBeenCalledExactlyOnceWith(TASKS[1].prompt);
    expect(selectRecipient).not.toHaveBeenCalled();
    expect(sources).toEqual([null]);
    expect(events("home_suggestion_clicked")).toEqual([
      expect.objectContaining({
        owner: "otto",
        expert_template_id: "template-bookkeeper",
      }),
    ]);
  });

  it("says that expert isn't on the team when the task has no expert name", async () => {
    const onSend = vi.fn<OnSend>(async () => "session-3");
    const { result } = await renderTasks({ onSend });

    act(() => result.current.sendTask(shownTask(result.current.tasks, "t3")));

    expect(
      await screen.findByText(
        "That expert isn't on your team, so Otto has this one.",
      ),
    ).toBeDefined();
    expect(onSend).toHaveBeenCalledExactlyOnceWith(TASKS[3].prompt);
  });

  it("says nothing about the team when no chat started", async () => {
    const onSend = vi.fn<OnSend>(async () => undefined);
    const { result } = await renderTasks({ onSend, experts: [] });

    act(() => result.current.sendTask(shownTask(result.current.tasks, "t1")));

    await waitFor(() => expect(onSend).toHaveBeenCalledOnce());
    await waitFor(() => expect(result.current.sendingId).toBeNull());
    expect(screen.queryByText(/isn't on your team/)).toBeNull();
  });

  it("says nothing about the team when the roster could not be read", async () => {
    // Owners are unknown then, not gone: the task still goes to Otto.
    const onSend = vi.fn<OnSend>(async () => "session-1");
    const { result } = await renderTasks({
      onSend,
      experts: [],
      hasRosterFailed: true,
    });

    act(() => result.current.sendTask(shownTask(result.current.tasks, "t1")));

    await waitFor(() => expect(writes).toHaveLength(1));
    expect(onSend).toHaveBeenCalledExactlyOnceWith(TASKS[1].prompt);
    expect(screen.queryByText(/isn't on your team/)).toBeNull();
  });

  it("shows no toast for a task that was always Otto's", async () => {
    const onSend = vi.fn<OnSend>(async () => "session-2");
    const { result } = await renderTasks({ onSend });

    act(() => result.current.sendTask(shownTask(result.current.tasks, "t2")));

    await waitFor(() => expect(result.current.sendingId).toBeNull());
    await waitFor(() => expect(writes).toHaveLength(1));
    expect(onSend).toHaveBeenCalledExactlyOnceWith(TASKS[2].prompt);
    expect(screen.queryByText(/isn't on your team/)).toBeNull();
  });
});

describe("useSuggestedTasks — what a click reports", () => {
  it("reports the owner, the template and the task's rank in the stored order", async () => {
    const selectRecipient = vi.fn();
    const { result, rerender, props } = await renderTasks({
      selectRecipient,
      recipientID: MARIA.id,
    });

    act(() => result.current.sendTask(shownTask(result.current.tasks, "t1")));
    await waitFor(() => expect(result.current.sendingId).toBeNull());

    act(() => result.current.sendTask(shownTask(result.current.tasks, "t2")));
    rerender({ ...props, recipientID: null });
    await waitFor(() => expect(writes).toHaveLength(2));

    expect(events("home_suggestion_clicked")).toEqual([
      {
        position: 0,
        surface: "home",
        remaining: 2,
        owner: "expert",
        expert_template_id: "template-bookkeeper",
        rank: 1,
      },
      {
        position: 1,
        surface: "home",
        remaining: 1,
        owner: "otto",
        expert_template_id: null,
        rank: 2,
      },
    ]);
  });
});

describe("useSuggestedTasks — a send that doesn't start a chat", () => {
  it("leaves the row open and writes nothing when no chat comes of the send", async () => {
    const onSend = vi.fn<OnSend>(async () => undefined);
    const { result } = await renderTasks({
      onSend,
      recipientID: MARIA.id,
    });

    act(() => result.current.sendTask(shownTask(result.current.tasks, "t1")));

    await waitFor(() => expect(result.current.sendingId).toBeNull());
    expect(onSend).toHaveBeenCalledExactlyOnceWith(TASKS[1].prompt);
    expect(client.isMutating()).toBe(0);
    expect(writes).toEqual([]);
    expect(shownTask(result.current.tasks, "t1").state).toBe("open");
    // The expert thread never happened, so the next one starts unlabelled.
    expect(takeExpertThreadSource()).toBeNull();
  });

  it("shows the failure toast and frees the row when the send fails", async () => {
    const onSend = vi.fn<OnSend>(async () => {
      throw new Error("The chat service is unavailable");
    });
    const { result } = await renderTasks({
      onSend,
      recipientID: MARIA.id,
    });

    act(() => result.current.sendTask(shownTask(result.current.tasks, "t1")));

    expect(await screen.findByText(FAILURE_TITLE)).toBeDefined();
    expect(
      screen.getByText(
        "The chat service is unavailable — it is still in your suggestions.",
      ),
    ).toBeDefined();
    await waitFor(() => expect(result.current.sendingId).toBeNull());
    expect(shownTask(result.current.tasks, "t1").state).toBe("open");
    expect(writes).toEqual([]);
    expect(takeExpertThreadSource()).toBeNull();
  });
});

describe("useSuggestedTasks — what it offers", () => {
  it("no longer offers a way to hide every task at once", async () => {
    const { result } = await renderTasks();

    expect(result.current).not.toHaveProperty("hideAll");
    expect(Object.keys(result.current).sort()).toEqual(
      [
        "allTasks",
        "experts",
        "expertsById",
        "isRosterLoaded",
        "isStored",
        "removeTask",
        "sendTask",
        "sendingId",
        "tasks",
      ].sort(),
    );
  });
});
