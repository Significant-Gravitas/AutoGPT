import { getGetBrainDumpIntroMockHandler200 } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump.msw";
import { getListExpertIdentitiesMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getGetHomeDashboardMockHandler200,
  getGetHomeDashboardResponseMock200,
} from "@/app/api/__generated__/endpoints/home/home.msw";
import type { ExpertIdentity } from "@/app/api/__generated__/models/expertIdentity";
import type { SuggestedTask } from "@/app/api/__generated__/models/suggestedTask";
import type { SuggestedTasksUpdate } from "@/app/api/__generated__/models/suggestedTasksUpdate";
import { Toaster } from "@/components/molecules/Toast/toaster";
import {
  TOUR_STARTING_POINT_ATTRIBUTE,
  TOUR_TARGET_ATTRIBUTE,
} from "@/lib/tour-targets";
import { server } from "@/mocks/mock-server";
import {
  act,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { withNuqsTestingAdapter } from "nuqs/adapters/testing";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { EmptySession } from "../../EmptySession/EmptySession";

const capture = vi.hoisted(() => vi.fn());
vi.mock("posthog-js", () => ({ default: { capture } }));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "user-1", user_metadata: { preferred_name: "Alex" } },
    isUserLoading: false,
    isLoggedIn: true,
  }),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: () => false,
    useFlagStatus: () => ({ enabled: false, ready: true }),
  };
});

const URL = "*/api/onboarding/brain-dump/suggested-tasks";
const COLLAPSED_KEY = "home.suggestedTasks.collapsed";
const SOMETHING_ELSE =
  "Want to do something else? Just write it in the textbox below.";

const MARIA: ExpertIdentity = {
  id: "expert-maria",
  name: "Maria",
  role: "Email & Lifecycle",
  avatar_url: null,
  is_archived: false,
};
const SAM: ExpertIdentity = {
  id: "expert-sam",
  name: "Sam",
  role: "Sales",
  avatar_url: null,
  is_archived: false,
};
// Fired: still in the roster the API returns, no longer on the team.
const JO: ExpertIdentity = {
  id: "expert-jo",
  name: "Jo",
  role: "Ops",
  avatar_url: null,
  is_archived: true,
};

const TASKS: SuggestedTask[] = [
  {
    id: "0",
    title: "Chase unpaid Shopify invoices",
    prompt: "Chase every unpaid Shopify invoice with a polite follow-up.",
    icon: "bell",
    state: "open",
    why: "Three invoices are more than 30 days late.",
    expert_id: MARIA.id,
    expert_template_id: "template-email",
    expert_name: MARIA.name,
  },
  {
    id: "1",
    title: "Turn sales calls into CRM updates",
    prompt: "After each sales call, update the CRM and draft a follow-up.",
    icon: "users",
    state: "open",
    why: null,
    expert_id: null,
    expert_template_id: null,
    expert_name: null,
  },
  {
    id: "2",
    title: "Watch competitor pricing",
    prompt: "Check competitor pricing weekly and tell me what moved.",
    icon: "target",
    state: "open",
    why: "Two rivals changed their prices last month.",
    expert_id: SAM.id,
    expert_template_id: "template-sales",
    expert_name: SAM.name,
  },
];

const MORE_TASKS: SuggestedTask[] = [
  {
    id: "3",
    title: "Draft the weekly newsletter",
    prompt: "Draft this week's newsletter from what we shipped.",
    state: "open",
    expert_id: null,
  },
  {
    id: "4",
    title: "Reconcile last month's expenses",
    prompt: "Match last month's card spend against the receipts.",
    state: "open",
    expert_id: null,
  },
];

const ALL_TASKS = [...TASKS, ...MORE_TASKS];

/** The stored list, which the POST handler changes as the server would. */
let stored: SuggestedTask[] = [];
let isStored = true;
let reads = 0;
let writes: SuggestedTasksUpdate[] = [];
/** Who each chat is with, as the server reads it for a started task. */
let chatExperts: Record<string, string | null> = {};

beforeEach(() => {
  capture.mockReset();
  window.localStorage.clear();
  stored = TASKS.map((task) => ({ ...task }));
  isStored = true;
  reads = 0;
  writes = [];
  chatExperts = {};
  server.use(
    http.get(URL, () => {
      reads += 1;
      return HttpResponse.json({ tasks: stored, stored: isStored });
    }),
    http.post(URL, async ({ request }) => {
      const update = (await request.json()) as SuggestedTasksUpdate;
      writes.push(update);
      stored = stored.map((task) => {
        if (!update.task_ids.includes(task.id)) return task;
        if (update.state === "open" && task.session_id) {
          return { ...task, state: "sent" };
        }
        const sessionID = update.session_id ?? task.session_id;
        return {
          ...task,
          state: update.state,
          session_id: sessionID,
          expert_id:
            sessionID && sessionID in chatExperts
              ? chatExperts[sessionID]
              : task.expert_id,
        };
      });
      isStored = true;
      return HttpResponse.json({ tasks: stored, stored: true });
    }),
    getListExpertIdentitiesMockHandler([MARIA, SAM, JO]),
    getGetHomeDashboardMockHandler200(
      getGetHomeDashboardResponseMock200({
        timezone: "UTC",
        attention: [],
        active_tasks: [],
        upcoming_tasks: [],
        agents: [],
      }),
    ),
  );
});

afterEach(() => {
  vi.useRealTimers();
});

function withGreeting() {
  server.use(
    getGetBrainDumpIntroMockHandler200({
      path: "A",
      greeting: "You have an invoicing problem and a sales problem.",
      prompts: [],
      greeting_done: false,
      greeting_pending: false,
      team: null,
      team_pending: false,
    }),
  );
}

function renderHome(
  args: {
    searchParams?: string;
    onStartTask?: (message: string) => Promise<string | undefined>;
  } = {},
) {
  const Wrapper = withNuqsTestingAdapter({
    searchParams: args.searchParams ?? "",
    hasMemory: true,
  });
  return render(
    <Wrapper>
      <EmptySession
        isCreatingSession={false}
        onCreateSession={() => {}}
        onSend={vi.fn()}
        onStartTask={args.onStartTask ?? vi.fn(async () => "session-1")}
      />
      <Toaster />
    </Wrapper>,
  );
}

function events(name: string) {
  return capture.mock.calls
    .filter(([event]) => event === name)
    .map(([, properties]) => properties);
}

function startName(owner: string, task: SuggestedTask) {
  return `Start with ${owner}: ${task.title}`;
}

function chatName(owner: string, task: SuggestedTask) {
  return `Open the chat with ${owner}: ${task.title}`;
}

function removeName(task: SuggestedTask) {
  return `Remove: ${task.title}`;
}

async function findBlock() {
  return screen.findByRole("region", { name: "Suggested first tasks" });
}

/** Lets the list's answer render, so an absence is not just "not loaded yet". */
async function settleList() {
  await waitFor(() => expect(reads).toBe(1));
  await act(async () => {});
}

function rows() {
  return screen.queryAllByTestId("suggested-task");
}

function titleOf(row: HTMLElement) {
  return ALL_TASKS.find((task) => row.textContent?.includes(task.title))?.title;
}

function rowOf(task: SuggestedTask) {
  const row = rows().find((candidate) => titleOf(candidate) === task.title);
  if (!row) throw new Error(`No row for "${task.title}"`);
  return row;
}

/** The titles of the rows carrying the "Start here" pill and the tint. */
function startHere() {
  return {
    pill: rows()
      .filter((row) => within(row).queryByText("Start here"))
      .map(titleOf),
    tint: rows()
      .filter((row) => row.classList.contains("bg-accent-surface"))
      .map(titleOf),
  };
}

/** The innermost element reading exactly `text`, across nested spans. */
function textIs(text: string) {
  return (_content: string, element: Element | null) =>
    element?.textContent === text &&
    Array.from(element.children).every((child) => child.textContent !== text);
}

function controlled(toggle: HTMLElement) {
  return document.getElementById(toggle.getAttribute("aria-controls") ?? "");
}

describe("Suggested first tasks on Home — who takes each task", () => {
  it("shows each row's owner with their avatar and role, and Otto for his own and for an expert not on the team", async () => {
    stored[2] = { ...stored[2], expert_id: "expert-gone" };
    renderHome();

    await screen.findByRole("button", { name: startName("Maria", TASKS[0]) });
    const maria = rowOf(TASKS[0]);
    expect(
      within(maria)
        .getByRole("img", { name: "Maria, AI Expert" })
        .getAttribute("width"),
    ).toBe("28");
    expect(
      within(maria).getByText(textIs("Maria, Email Marketing Manager")),
    ).toBeDefined();

    for (const task of [TASKS[1], TASKS[2]]) {
      const row = rowOf(task);
      expect(
        within(row).getByRole("button", { name: startName("Otto", task) }),
      ).toBeDefined();
      expect(
        within(row)
          .getByRole("img", { name: "Otto, your personal Head of AI" })
          .getAttribute("width"),
      ).toBe("28");
      expect(within(row).getByText("Otto")).toBeDefined();
    }
  });

  it("sends an expert's task to that expert once the composer is addressed to them, then shows it in progress with them", async () => {
    let wasAddressedToMaria = false;
    const onStartTask = vi.fn(async () => {
      wasAddressedToMaria =
        screen.queryByPlaceholderText("What should Maria work on?") !== null;
      return "session-1";
    });
    renderHome({ onStartTask });

    await userEvent.click(
      await screen.findByRole("button", {
        name: startName("Maria", TASKS[0]),
      }),
    );

    const started = await screen.findByRole("link", {
      name: chatName("Maria", TASKS[0]),
    });
    expect(started.getAttribute("href")).toBe("/copilot?sessionId=session-1");
    expect(within(started).getByText("In progress with Maria")).toBeDefined();
    expect(onStartTask).toHaveBeenCalledExactlyOnceWith(TASKS[0].prompt);
    expect(wasAddressedToMaria).toBe(true);
    await waitFor(() =>
      expect(writes).toEqual([
        { task_ids: ["0"], state: "sent", session_id: "session-1" },
      ]),
    );
    expect(events("home_suggestion_clicked")).toEqual([
      {
        position: 0,
        surface: "home",
        remaining: 2,
        owner: "expert",
        expert_template_id: "template-email",
        rank: 0,
      },
    ]);
    expect(screen.queryByText(/isn't on your team/)).toBeNull();
  });

  it("gives Otto a task whose expert has left the team, says so, and shows it in progress with Otto", async () => {
    stored[0] = { ...stored[0], state: "dismissed" };
    stored[2] = {
      ...stored[2],
      expert_id: JO.id,
      expert_template_id: "template-ops",
      expert_name: JO.name,
    };
    chatExperts = { "session-1": null };
    renderHome();

    await screen.findByRole("button", { name: startName("Otto", TASKS[1]) });
    await userEvent.click(
      screen.getByRole("button", { name: startName("Otto", TASKS[2]) }),
    );

    expect(
      await screen.findByText("Jo isn't on your team, so Otto has this one."),
    ).toBeDefined();
    const started = await screen.findByRole("link", {
      name: chatName("Otto", TASKS[2]),
    });
    expect(within(started).getByText("In progress with Otto")).toBeDefined();
    expect(events("home_suggestion_clicked")).toEqual([
      {
        position: 1,
        surface: "home",
        remaining: 1,
        owner: "otto",
        expert_template_id: "template-ops",
        rank: 2,
      },
    ]);
  });

  it("sends Otto's own task to Otto even while the composer is addressed to Maria, without the left-the-team note", async () => {
    let wasAddressedToMaria = true;
    const onStartTask = vi.fn(async () => {
      wasAddressedToMaria =
        screen.queryByPlaceholderText("What should Maria work on?") !== null;
      return "session-2";
    });
    renderHome({ searchParams: "?expertId=expert-maria", onStartTask });

    await screen.findByPlaceholderText("What should Maria work on?");
    const block = await findBlock();
    await userEvent.click(
      await within(block).findByRole("button", {
        name: startName("Otto", TASKS[1]),
      }),
    );

    expect(
      await screen.findByRole("link", { name: chatName("Otto", TASKS[1]) }),
    ).toBeDefined();
    expect(onStartTask).toHaveBeenCalledExactlyOnceWith(TASKS[1].prompt);
    expect(wasAddressedToMaria).toBe(false);
    expect(screen.queryByText(/isn't on your team/)).toBeNull();
  });

  it("keeps naming the expert a started task went to after that expert is fired", async () => {
    // The server names the expert the task's chat is with, fired or not.
    stored[0] = {
      ...stored[0],
      state: "sent",
      session_id: "session-jo",
      expert_id: JO.id,
    };
    renderHome();

    const link = await screen.findByRole("link", {
      name: chatName("Jo", TASKS[0]),
    });
    expect(link.getAttribute("href")).toContain("session-jo");
    expect(
      within(rowOf(TASKS[0])).getByText("In progress with Jo"),
    ).toBeDefined();
    expect(
      screen.queryByRole("button", { name: startName("Jo", TASKS[0]) }),
    ).toBeNull();
  });
});

describe("Suggested first tasks on Home — the rows", () => {
  it("marks only the first open row Start here, and moves it to the next open row once that one is started", async () => {
    renderHome();
    await screen.findByRole("button", { name: startName("Maria", TASKS[0]) });

    expect(startHere()).toEqual({
      pill: [TASKS[0].title],
      tint: [TASKS[0].title],
    });

    await userEvent.click(
      screen.getByRole("button", { name: startName("Maria", TASKS[0]) }),
    );
    await screen.findByRole("link", { name: chatName("Maria", TASKS[0]) });

    await waitFor(() =>
      expect(startHere()).toEqual({
        pill: [TASKS[1].title],
        tint: [TASKS[1].title],
      }),
    );
  });

  it("moves Start here to the next open row when the first is removed", async () => {
    renderHome();

    await userEvent.click(
      await screen.findByRole("button", { name: removeName(TASKS[0]) }),
    );

    await waitFor(() => expect(rows()).toHaveLength(2));
    expect(startHere()).toEqual({
      pill: [TASKS[1].title],
      tint: [TASKS[1].title],
    });
  });

  it("shows a started task as a link to its chat, in progress with its owner, with no Remove button", async () => {
    stored[0] = { ...stored[0], state: "sent", session_id: "chat /one?" };
    renderHome();

    const started = await screen.findByRole("link", {
      name: chatName("Maria", TASKS[0]),
    });

    expect(started.getAttribute("href")).toBe(
      "/copilot?sessionId=chat%20%2Fone%3F",
    );
    expect(within(started).getByText("In progress with Maria")).toBeDefined();
    const row = rowOf(TASKS[0]);
    expect(within(row).queryByRole("button")).toBeNull();
    expect(
      screen.queryByRole("button", { name: removeName(TASKS[0]) }),
    ).toBeNull();
    expect(startHere()).toEqual({
      pill: [TASKS[1].title],
      tint: [TASKS[1].title],
    });
  });

  it("keeps a started row listed when Home is opened again", async () => {
    const view = renderHome();
    await userEvent.click(
      await screen.findByRole("button", { name: startName("Otto", TASKS[1]) }),
    );
    await screen.findByRole("link", { name: chatName("Otto", TASKS[1]) });
    await waitFor(() => expect(stored[1].state).toBe("sent"));

    view.unmount();
    renderHome();

    expect(
      await screen.findByRole("link", { name: chatName("Otto", TASKS[1]) }),
    ).toBeDefined();
    await waitFor(() => expect(reads).toBeGreaterThanOrEqual(2));
    expect(rows().map(titleOf)).toEqual(TASKS.map((task) => task.title));
  });

  it("shows a task's why under its title, and nothing there when it has none", async () => {
    renderHome();
    await screen.findByRole("button", { name: startName("Sam", TASKS[2]) });

    expect(within(rowOf(TASKS[0])).getByText(TASKS[0].why ?? "")).toBeDefined();
    expect(within(rowOf(TASKS[2])).getByText(TASKS[2].why ?? "")).toBeDefined();
    expect(rowOf(TASKS[1]).textContent).toBe(`${TASKS[1].title}Otto`);
  });

  it("marks the open rows' buttons, and only those, as where the walkthrough starts", async () => {
    stored[0] = { ...stored[0], state: "sent", session_id: "session-1" };
    renderHome();

    const started = await screen.findByRole("link", {
      name: chatName("Maria", TASKS[0]),
    });
    const openButtons = [
      await screen.findByRole("button", { name: startName("Otto", TASKS[1]) }),
      await screen.findByRole("button", { name: startName("Sam", TASKS[2]) }),
    ];

    expect(
      Array.from(
        document.querySelectorAll(`[${TOUR_STARTING_POINT_ATTRIBUTE}]`),
      ),
    ).toEqual(openButtons);
    expect(started.hasAttribute(TOUR_STARTING_POINT_ATTRIBUTE)).toBe(false);
  });
});

describe("Suggested first tasks on Home — removing a task", () => {
  it("removes a row in one write, says Task removed, and Undo puts it back", async () => {
    renderHome();

    await userEvent.click(
      await screen.findByRole("button", { name: removeName(TASKS[1]) }),
    );

    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: startName("Otto", TASKS[1]) }),
      ).toBeNull(),
    );
    await waitFor(() =>
      expect(writes).toEqual([{ task_ids: ["1"], state: "dismissed" }]),
    );
    expect(events("home_suggestion_dismissed")).toEqual([
      { position: 1, surface: "home", remaining: 2, scope: "one" },
    ]);

    await screen.findByText("Task removed");
    await userEvent.click(screen.getByRole("button", { name: "Undo" }));

    expect(
      await screen.findByRole("button", { name: startName("Otto", TASKS[1]) }),
    ).toBeDefined();
    await waitFor(() =>
      expect(writes.at(-1)).toEqual({ task_ids: ["1"], state: "open" }),
    );
    expect(writes).toHaveLength(2);
    expect(events("home_suggestion_dismissed")).toHaveLength(1);
  });

  it("keeps the row removed once the five seconds for Undo have passed", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    renderHome();

    await user.click(
      await screen.findByRole("button", { name: removeName(TASKS[1]) }),
    );
    await screen.findByText("Task removed");

    await act(async () => {
      vi.advanceTimersByTime(5000);
    });

    await waitFor(() => expect(screen.queryByText("Task removed")).toBeNull());
    expect(screen.queryByRole("button", { name: "Undo" })).toBeNull();
    expect(
      screen.queryByRole("button", { name: startName("Otto", TASKS[1]) }),
    ).toBeNull();
    expect(writes).toEqual([{ task_ids: ["1"], state: "dismissed" }]);
  });

  it("keeps a removed task off Home when it is opened again", async () => {
    stored[2] = { ...stored[2], state: "dismissed" };
    renderHome();

    const block = await findBlock();

    expect(within(block).getAllByTestId("suggested-task")).toHaveLength(2);
    expect(
      within(block).queryByRole("button", { name: startName("Sam", TASKS[2]) }),
    ).toBeNull();
  });

  it("offers no way to hide every task at once", async () => {
    renderHome();
    await findBlock();

    expect(screen.queryByRole("button", { name: "Hide all" })).toBeNull();
    expect(screen.queryByText("Hide all")).toBeNull();
  });
});

describe("Suggested first tasks on Home — Hide and Show", () => {
  it("folds the list away with Hide, counting the tasks not removed, and brings it back with Show", async () => {
    stored[0] = { ...stored[0], state: "sent", session_id: "session-1" };
    stored[2] = { ...stored[2], state: "dismissed" };
    renderHome();
    const block = await findBlock();

    const hide = within(block).getByRole("button", { name: "Hide" });
    expect(hide.getAttribute("aria-expanded")).toBe("true");
    expect(
      controlled(hide)?.contains(
        screen.getByRole("list", { name: "Your first tasks" }),
      ),
    ).toBe(true);

    await userEvent.click(hide);

    const show = await within(block).findByRole("button", {
      name: "2 tasks · Show",
    });
    expect(show.getAttribute("aria-expanded")).toBe("false");
    await waitFor(() =>
      expect(
        screen.queryByRole("list", { name: "Your first tasks" }),
      ).toBeNull(),
    );
    expect(controlled(show)?.textContent).toBe("");
    expect(events("home_suggestions_collapsed")).toEqual([{ open: 1 }]);
    expect(window.localStorage.getItem(COLLAPSED_KEY)).toBe(
      JSON.stringify(["user-1"]),
    );
    expect(writes).toEqual([]);

    await userEvent.click(show);

    expect(
      await screen.findByRole("list", { name: "Your first tasks" }),
    ).toBeDefined();
    const hideAgain = within(block).getByRole("button", { name: "Hide" });
    expect(hideAgain.getAttribute("aria-expanded")).toBe("true");
    expect(events("home_suggestions_expanded")).toEqual([{ open: 1 }]);
    expect(window.localStorage.getItem(COLLAPSED_KEY)).toBe("[]");
    expect(writes).toEqual([]);
  });

  it("stays folded away when Home is opened again", async () => {
    const view = renderHome();
    const block = await findBlock();
    await userEvent.click(within(block).getByRole("button", { name: "Hide" }));
    await within(block).findByRole("button", { name: "3 tasks · Show" });

    view.unmount();
    renderHome();

    const reopened = await findBlock();
    const show = within(reopened).getByRole("button", {
      name: "3 tasks · Show",
    });
    expect(show.getAttribute("aria-expanded")).toBe("false");
    expect(rows()).toEqual([]);
  });

  it("opens the list for a user even when another account on this browser folded theirs away", async () => {
    window.localStorage.setItem(COLLAPSED_KEY, JSON.stringify(["user-2"]));
    renderHome();

    const block = await findBlock();

    expect(
      within(block)
        .getByRole("button", { name: "Hide" })
        .getAttribute("aria-expanded"),
    ).toBe("true");
    expect(within(block).getAllByTestId("suggested-task")).toHaveLength(3);
  });
});

describe("Suggested first tasks on Home — more tasks", () => {
  it("keeps the rows after the first three under a closed '2 more tasks' row that opens on click", async () => {
    stored = ALL_TASKS.map((task) => ({ ...task }));
    renderHome();

    const more = await screen.findByRole("button", { name: "2 more tasks" });
    expect(more.getAttribute("aria-expanded")).toBe("false");
    expect(controlled(more)?.hidden).toBe(true);
    expect(rows().map(titleOf)).toEqual(TASKS.map((task) => task.title));

    const firstTasks = screen.getByRole("list", { name: "Your first tasks" });
    expect(firstTasks.getAttribute(TOUR_TARGET_ATTRIBUTE)).toBe("first-tasks");
    expect(within(firstTasks).getAllByRole("listitem")).toHaveLength(3);

    await userEvent.click(more);

    expect(
      await screen.findByRole("button", {
        name: startName("Otto", MORE_TASKS[0]),
      }),
    ).toBeDefined();
    expect(more.getAttribute("aria-expanded")).toBe("true");
    expect(controlled(more)?.hidden).toBe(false);
    expect(
      within(screen.getByRole("list", { name: "More tasks" }))
        .getAllByTestId("suggested-task")
        .map(titleOf),
    ).toEqual(MORE_TASKS.map((task) => task.title));
    expect(within(firstTasks).getAllByRole("listitem")).toHaveLength(3);
    expect(within(firstTasks).queryByText(MORE_TASKS[0].title)).toBeNull();
  });

  it("opens the more tasks by itself once one of the first three is removed", async () => {
    stored = ALL_TASKS.map((task) => ({ ...task }));
    renderHome();

    expect(
      (
        await screen.findByRole("button", { name: "2 more tasks" })
      ).getAttribute("aria-expanded"),
    ).toBe("false");
    await userEvent.click(
      screen.getByRole("button", { name: removeName(TASKS[1]) }),
    );

    expect(
      await screen.findByRole("button", {
        name: startName("Otto", MORE_TASKS[0]),
      }),
    ).toBeDefined();
    expect(
      screen
        .getByRole("button", { name: "2 more tasks" })
        .getAttribute("aria-expanded"),
    ).toBe("true");
  });

  it("opens the more tasks by itself once one of the first three is started", async () => {
    stored = ALL_TASKS.map((task) => ({ ...task }));
    renderHome();

    await userEvent.click(
      await screen.findByRole("button", { name: startName("Otto", TASKS[1]) }),
    );

    await screen.findByRole("link", { name: chatName("Otto", TASKS[1]) });
    expect(
      await screen.findByRole("button", {
        name: startName("Otto", MORE_TASKS[1]),
      }),
    ).toBeDefined();
    expect(
      screen
        .getByRole("button", { name: "2 more tasks" })
        .getAttribute("aria-expanded"),
    ).toBe("true");
  });

  it("moves Start here into the more tasks once the first three are all started", async () => {
    stored = ALL_TASKS.map((task, index) =>
      index < 3
        ? { ...task, state: "sent", session_id: `session-${index}` }
        : { ...task },
    );
    renderHome();

    await screen.findByRole("button", {
      name: startName("Otto", MORE_TASKS[0]),
    });

    expect(startHere()).toEqual({
      pill: [MORE_TASKS[0].title],
      tint: [MORE_TASKS[0].title],
    });
  });

  it("shows no more-tasks row with three tasks, nor when every extra task was removed", async () => {
    const view = renderHome();
    await settleList();
    await findBlock();

    expect(screen.queryByRole("button", { name: /more tasks?$/ })).toBeNull();

    view.unmount();
    stored = [
      ...TASKS.map((task) => ({ ...task })),
      { ...MORE_TASKS[0], state: "dismissed" },
    ];
    reads = 0;
    renderHome();
    await settleList();
    await findBlock();

    expect(screen.queryByRole("button", { name: /more tasks?$/ })).toBeNull();
  });

  it("says '1 more task' for a single extra task", async () => {
    stored = [...TASKS, MORE_TASKS[0]].map((task) => ({ ...task }));
    renderHome();

    expect(
      await screen.findByRole("button", { name: "1 more task" }),
    ).toBeDefined();
  });
});

describe("Suggested first tasks on Home — when to show them", () => {
  it("leaves the generic starters off Home for a user who never got a list", async () => {
    isStored = false;
    renderHome();

    await settleList();

    expect(
      screen.queryByRole("region", { name: "Suggested first tasks" }),
    ).toBeNull();
  });

  it("shows the list, without the team row, while an expert is the recipient", async () => {
    renderHome({ searchParams: "?expertId=expert-maria" });

    await screen.findByPlaceholderText("What should Maria work on?");
    const block = await findBlock();

    expect(within(block).getAllByTestId("suggested-task")).toHaveLength(3);
    expect(screen.queryByRole("list", { name: "Your team" })).toBeNull();
  });
});

describe("Suggested first tasks — the greeting page", () => {
  it("shows the tasks as the list, with Otto and the hired experts under Your team, above the composer", async () => {
    withGreeting();
    renderHome();

    const card = await screen.findByTestId("onboarding-intro-card");
    const team = await within(card).findByRole("list", { name: "Your team" });
    await waitFor(() =>
      expect(
        within(team)
          .getAllByRole("listitem")
          .map((chip) => chip.textContent),
      ).toEqual([
        "OttoHead of AI",
        "MariaEmail Marketing Manager",
        "SamSales Development Rep",
      ]),
    );

    const list = within(card).getByRole("list", { name: "Your first tasks" });
    expect(within(list).getAllByRole("listitem")).toHaveLength(3);
    expect(
      within(list).getByRole("button", { name: startName("Maria", TASKS[0]) }),
    ).toBeDefined();
    expect(
      within(list).getByRole("button", { name: startName("Otto", TASKS[1]) }),
    ).toBeDefined();
    expect(
      within(list).getByRole("button", { name: startName("Sam", TASKS[2]) }),
    ).toBeDefined();
    expect(within(card).getByText(SOMETHING_ELSE)).toBeDefined();
    expect(
      screen.getByRole("textbox", { name: "Chat message input" }),
    ).toBeDefined();
    expect(screen.queryByText("Hide all")).toBeNull();
  });

  it("folds the list and the 'something else' line away with Hide, and keeps the composer", async () => {
    withGreeting();
    renderHome();

    const card = await screen.findByTestId("onboarding-intro-card");
    await within(card).findByText(SOMETHING_ELSE);
    await userEvent.click(within(card).getByRole("button", { name: "Hide" }));

    const show = await within(card).findByRole("button", {
      name: "3 tasks · Show",
    });
    expect(show.getAttribute("aria-expanded")).toBe("false");
    await waitFor(() =>
      expect(
        screen.queryByRole("list", { name: "Your first tasks" }),
      ).toBeNull(),
    );
    expect(within(card).queryByText(SOMETHING_ELSE)).toBeNull();
    expect(
      screen.getByRole("textbox", { name: "Chat message input" }),
    ).toBeDefined();
    expect(events("home_suggestions_collapsed")).toEqual([{ open: 3 }]);
  });

  it("sends the full prompt and records which chat it started", async () => {
    withGreeting();
    const onStartTask = vi.fn(async () => "session-1");
    renderHome({ onStartTask });

    await userEvent.click(
      await screen.findByRole("button", { name: startName("Otto", TASKS[1]) }),
    );

    expect(
      await screen.findByRole("link", { name: chatName("Otto", TASKS[1]) }),
    ).toBeDefined();
    expect(onStartTask).toHaveBeenCalledExactlyOnceWith(TASKS[1].prompt);
    await waitFor(() =>
      expect(writes).toEqual([
        { task_ids: ["1"], state: "sent", session_id: "session-1" },
      ]),
    );
    expect(events("home_suggestion_clicked")).toEqual([
      {
        position: 1,
        surface: "intro",
        remaining: 2,
        owner: "otto",
        expert_template_id: null,
        rank: 1,
      },
    ]);
  });

  it("removes a greeting task and restores it with Undo", async () => {
    withGreeting();
    renderHome();

    await userEvent.click(
      await screen.findByRole("button", { name: removeName(TASKS[1]) }),
    );

    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: startName("Otto", TASKS[1]) }),
      ).toBeNull(),
    );
    await waitFor(() =>
      expect(writes).toEqual([{ task_ids: ["1"], state: "dismissed" }]),
    );
    expect(events("home_suggestion_dismissed")).toEqual([
      { position: 1, surface: "intro", remaining: 2, scope: "one" },
    ]);

    await screen.findByText("Task removed");
    await userEvent.click(screen.getByRole("button", { name: "Undo" }));

    expect(
      await screen.findByRole("button", { name: startName("Otto", TASKS[1]) }),
    ).toBeDefined();
    await waitFor(() =>
      expect(writes.at(-1)).toEqual({ task_ids: ["1"], state: "open" }),
    );
  });

  it("locks the rows while a task starts, then unlocks them after failure", async () => {
    withGreeting();
    let rejectStart = (_error: Error) => {};
    const pending = new Promise<string>((_resolve, reject) => {
      rejectStart = reject;
    });
    const onStartTask = vi.fn(() => pending);
    renderHome({ onStartTask });
    await screen.findByRole("button", { name: startName("Sam", TASKS[2]) });
    await userEvent.click(
      screen.getByRole("button", { name: startName("Otto", TASKS[1]) }),
    );

    try {
      await waitFor(() => expect(onStartTask).toHaveBeenCalledOnce());
      for (const row of rows()) {
        for (const button of within(row).getAllByRole("button")) {
          expect(button).toHaveProperty("disabled", true);
        }
      }
      await userEvent.click(
        screen.getByRole("button", { name: startName("Sam", TASKS[2]) }),
      );
      await userEvent.click(
        screen.getByRole("button", { name: removeName(TASKS[2]) }),
      );
      expect(onStartTask).toHaveBeenCalledOnce();
      expect(writes).toEqual([]);
    } finally {
      await act(async () => rejectStart(new Error("offline")));
    }

    await screen.findByText("Could not start that task");
    for (const row of rows()) {
      for (const button of within(row).getAllByRole("button")) {
        expect(button).toHaveProperty("disabled", false);
      }
    }
  });

  it("retries a failed sent write with the same chat instead of reopening the row", async () => {
    withGreeting();
    let failures = 1;
    server.use(
      http.post(URL, async ({ request }) => {
        const update = (await request.json()) as SuggestedTasksUpdate;
        writes.push(update);
        if (failures > 0) {
          failures -= 1;
          return HttpResponse.json({ detail: "busy" }, { status: 503 });
        }
        stored = stored.map((task) =>
          update.task_ids.includes(task.id)
            ? { ...task, state: update.state, session_id: update.session_id }
            : task,
        );
        return HttpResponse.json({ tasks: stored, stored: true });
      }),
    );
    renderHome();

    await userEvent.click(
      await screen.findByRole("button", { name: startName("Otto", TASKS[1]) }),
    );

    const sent = { task_ids: ["1"], state: "sent", session_id: "session-1" };
    await waitFor(() => expect(writes).toEqual([sent, sent]), {
      timeout: 4000,
    });
    expect(stored[1].state).toBe("sent");
    expect(
      screen.getByRole("link", { name: chatName("Otto", TASKS[1]) }),
    ).toBeDefined();
  });

  it("keeps a row open when its send never started a chat", async () => {
    withGreeting();
    renderHome({
      onStartTask: vi.fn(async () => {
        throw new Error("network down");
      }),
    });

    await userEvent.click(
      await screen.findByRole("button", { name: startName("Otto", TASKS[1]) }),
    );

    await screen.findByText("Could not start that task");
    expect(writes).toEqual([]);
    expect(
      screen.getByRole("button", { name: startName("Otto", TASKS[1]) }),
    ).toHaveProperty("disabled", false);
  });
});

describe("Suggested first tasks — while the greeting is being written", () => {
  it("shows the personalised list once the greeting lands, never the generic set before it", async () => {
    const generic: SuggestedTask[] = [
      {
        id: "0",
        title: "Get a sourced research brief on any topic in minutes",
        prompt: "I need research support.",
        icon: "magnifying-glass",
        state: "open",
      },
    ];
    // The pipeline writes the greeting and the personalised list together.
    let introReads = 0;
    let written = false;
    server.use(
      http.get("*/api/onboarding/brain-dump/intro", () => {
        introReads += 1;
        written = introReads > 1;
        return HttpResponse.json({
          path: "A",
          greeting: written ? "You have an invoicing problem." : "",
          prompts: [],
          greeting_done: false,
          greeting_pending: !written,
          team: null,
          team_pending: false,
        });
      }),
      http.get(URL, () =>
        HttpResponse.json(
          written
            ? { tasks: stored, stored: true }
            : { tasks: generic, stored: false },
        ),
      ),
    );
    renderHome();

    expect(
      await screen.findByRole(
        "button",
        { name: startName("Otto", TASKS[1]) },
        { timeout: 5000 },
      ),
    ).toBeDefined();
    expect(screen.queryByText(generic[0].title)).toBeNull();
  });
});

describe("Suggested first tasks — keyboard and screen readers", () => {
  function description(control: HTMLElement) {
    const ids = control.getAttribute("aria-describedby");
    if (!ids) return null;
    return ids
      .split(" ")
      .map((id) => document.getElementById(id)?.textContent)
      .join(" ");
  }

  it("moves focus to the next row when a task is removed, and keeps it there once Undo puts the task back", async () => {
    renderHome();

    await userEvent.click(
      await screen.findByRole("button", { name: removeName(TASKS[1]) }),
    );

    await waitFor(() =>
      expect(document.activeElement).toBe(
        screen.getByRole("button", { name: startName("Sam", TASKS[2]) }),
      ),
    );

    await screen.findByText("Task removed");
    await userEvent.click(screen.getByRole("button", { name: "Undo" }));

    await screen.findByRole("button", { name: startName("Otto", TASKS[1]) });
    // The toast hands focus back to where it was before the Undo.
    await waitFor(() =>
      expect(document.activeElement).toBe(
        screen.getByRole("button", { name: startName("Sam", TASKS[2]) }),
      ),
    );
  });

  it("moves focus to the row before when the last row is removed", async () => {
    renderHome();

    await userEvent.click(
      await screen.findByRole("button", { name: removeName(TASKS[2]) }),
    );

    await waitFor(() =>
      expect(document.activeElement).toBe(
        screen.getByRole("button", { name: startName("Otto", TASKS[1]) }),
      ),
    );
  });

  it("keeps the more tasks open through an Undo and through Hide and Show", async () => {
    stored = ALL_TASKS.map((task) => ({ ...task }));
    renderHome();
    const block = await findBlock();

    await userEvent.click(
      await screen.findByRole("button", { name: removeName(TASKS[1]) }),
    );
    await screen.findByText("Task removed");
    await userEvent.click(screen.getByRole("button", { name: "Undo" }));
    await screen.findByRole("button", { name: startName("Otto", TASKS[1]) });

    const more = screen.getByRole("button", { name: "2 more tasks" });
    expect(more.getAttribute("aria-expanded")).toBe("true");

    await userEvent.click(within(block).getByRole("button", { name: "Hide" }));
    await userEvent.click(
      await within(block).findByRole("button", { name: "5 tasks · Show" }),
    );

    expect(
      (
        await screen.findByRole("button", { name: "2 more tasks" })
      ).getAttribute("aria-expanded"),
    ).toBe("true");
    expect(
      screen.getByRole("button", { name: startName("Otto", MORE_TASKS[0]) }),
    ).toBeDefined();
  });

  it("describes each row with its Start here pill and its why, and says which task is starting", async () => {
    let finishStart = (_session: string) => {};
    const pending = new Promise<string>((resolve) => {
      finishStart = resolve;
    });
    renderHome({ onStartTask: vi.fn(() => pending) });

    const first = await screen.findByRole("button", {
      name: startName("Maria", TASKS[0]),
    });
    expect(description(first)).toBe(
      "Start here Three invoices are more than 30 days late.",
    );
    expect(
      description(
        screen.getByRole("button", { name: startName("Otto", TASKS[1]) }),
      ),
    ).toBeNull();
    expect(
      description(
        screen.getByRole("button", { name: startName("Sam", TASKS[2]) }),
      ),
    ).toBe("Two rivals changed their prices last month.");

    await userEvent.click(
      screen.getByRole("button", { name: startName("Otto", TASKS[1]) }),
    );

    try {
      const status = await screen.findByText(`Starting: ${TASKS[1].title}`);
      expect(status.getAttribute("role")).toBe("status");
    } finally {
      await act(async () => finishStart("session-1"));
    }
  });
});
