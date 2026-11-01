import { getGetBrainDumpIntroMockHandler200 } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump.msw";
import { getListExpertIdentitiesMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { SuggestedTask } from "@/app/api/__generated__/models/suggestedTask";
import type { SuggestedTasksUpdate } from "@/app/api/__generated__/models/suggestedTasksUpdate";
import { Toaster } from "@/components/molecules/Toast/toaster";
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
import { beforeEach, describe, expect, it, vi } from "vitest";
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

const TASKS: SuggestedTask[] = [
  {
    id: "0",
    title: "Chase unpaid Shopify invoices",
    prompt: "Chase every unpaid Shopify invoice with a polite follow-up.",
    icon: "bell",
    state: "open",
  },
  {
    id: "1",
    title: "Turn sales calls into CRM updates",
    prompt: "After each sales call, update the CRM and draft a follow-up.",
    icon: "users",
    state: "open",
  },
  {
    id: "2",
    title: "Watch competitor pricing",
    prompt: "Check competitor pricing weekly and tell me what moved.",
    icon: "target",
    state: "open",
  },
];

/** The stored list, which the POST handler changes as the server would. */
let stored: SuggestedTask[] = [];
let isStored = true;
let reads = 0;
let writes: SuggestedTasksUpdate[] = [];

beforeEach(() => {
  capture.mockReset();
  window.localStorage.clear();
  stored = TASKS.map((task) => ({ ...task }));
  isStored = true;
  reads = 0;
  writes = [];
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
        return {
          ...task,
          state: update.state,
          session_id: update.session_id ?? task.session_id,
        };
      });
      isStored = true;
      return HttpResponse.json({ tasks: stored, stored: true });
    }),
    getListExpertIdentitiesMockHandler([]),
  );
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

async function findBlock() {
  return screen.findByRole("region", { name: "Suggested first tasks" });
}

/** Lets the list's answer render, so an absence is not just "not loaded yet". */
async function settleList() {
  await waitFor(() => expect(reads).toBe(1));
  await act(async () => {});
}

describe("Suggested first tasks — the greeting page", () => {
  it("hides a greeting card and restores it with Undo", async () => {
    withGreeting();
    renderHome();

    await userEvent.click(
      await screen.findByRole("button", { name: `Hide: ${TASKS[1].title}` }),
    );

    await waitFor(() =>
      expect(screen.queryByRole("button", { name: TASKS[1].title })).toBeNull(),
    );
    await waitFor(() =>
      expect(writes).toEqual([{ task_ids: ["1"], state: "dismissed" }]),
    );
    expect(events("home_suggestion_dismissed")).toEqual([
      { position: 1, surface: "intro", remaining: 2, scope: "one" },
    ]);

    await userEvent.click(await screen.findByRole("button", { name: "Undo" }));

    expect(
      await screen.findByRole("button", { name: TASKS[1].title }),
    ).toBeDefined();
    await waitFor(() =>
      expect(writes.at(-1)).toEqual({ task_ids: ["1"], state: "open" }),
    );
  });

  it("links a started greeting task to its chat", async () => {
    withGreeting();
    stored[0] = { ...stored[0], state: "sent", session_id: "chat /one?" };
    renderHome();

    const started = await screen.findByRole("link", {
      name: `${TASKS[0].title} In progress`,
    });

    expect(started.getAttribute("href")).toBe(
      "/copilot?sessionId=chat%20%2Fone%3F",
    );
    expect(within(started).getByText("In progress")).toBeDefined();
    expect(screen.queryByRole("button", { name: TASKS[0].title })).toBeNull();
  });

  it("locks the cards while a task starts, then unlocks them after failure", async () => {
    withGreeting();
    let rejectStart = (_error: Error) => {};
    const pending = new Promise<string>((_resolve, reject) => {
      rejectStart = reject;
    });
    const onStartTask = vi.fn(() => pending);
    renderHome({ onStartTask });
    await userEvent.click(
      await screen.findByRole("button", { name: TASKS[0].title }),
    );

    try {
      expect(await screen.findByLabelText("Starting")).toBeDefined();
      for (const card of screen.getAllByTestId("suggested-task")) {
        for (const button of within(card).getAllByRole("button")) {
          expect(button).toHaveProperty("disabled", true);
        }
      }
      await userEvent.click(
        screen.getByRole("button", { name: TASKS[1].title }),
      );
      await userEvent.click(
        screen.getByRole("button", { name: `Hide: ${TASKS[1].title}` }),
      );
      expect(onStartTask).toHaveBeenCalledOnce();
      expect(writes).toEqual([]);
    } finally {
      await act(async () => rejectStart(new Error("offline")));
    }

    await screen.findByText("Could not start that task");
    for (const task of TASKS) {
      expect(screen.getByRole("button", { name: task.title })).toHaveProperty(
        "disabled",
        false,
      );
      expect(
        screen.getByRole("button", { name: `Hide: ${task.title}` }),
      ).toHaveProperty("disabled", false);
    }
  });

  it("shows the named tools in title order without changing the send label", async () => {
    withGreeting();
    stored[0] = {
      ...stored[0],
      title: "Summarize Slack, GitHub, and Linear updates in Notion",
    };
    renderHome();

    const card = await screen.findByRole("button", { name: stored[0].title });
    expect(
      Array.from(card.querySelectorAll("img")).map((logo) =>
        logo.getAttribute("src"),
      ),
    ).toEqual([
      "/integrations/slack.png",
      "/integrations/github.png",
      "/integrations/linear.png",
    ]);
  });

  it("sends the full prompt and records which chat it started", async () => {
    withGreeting();
    const onStartTask = vi.fn(async () => "session-1");
    renderHome({ onStartTask });

    await userEvent.click(
      await screen.findByRole("button", { name: TASKS[0].title }),
    );

    expect(onStartTask).toHaveBeenCalledExactlyOnceWith(TASKS[0].prompt);
    await waitFor(() =>
      expect(writes).toEqual([
        { task_ids: ["0"], state: "sent", session_id: "session-1" },
      ]),
    );
    expect(events("home_suggestion_clicked")).toEqual([
      { position: 0, surface: "intro", remaining: 2 },
    ]);
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
      await screen.findByRole("button", { name: TASKS[0].title }),
    );

    const sent = { task_ids: ["0"], state: "sent", session_id: "session-1" };
    await waitFor(() => expect(writes).toEqual([sent, sent]), {
      timeout: 4000,
    });
    expect(stored[0].state).toBe("sent");
  });

  it("keeps a row open when its send never started a chat", async () => {
    withGreeting();
    renderHome({
      onStartTask: vi.fn(async () => {
        throw new Error("network down");
      }),
    });

    await userEvent.click(
      await screen.findByRole("button", { name: TASKS[1].title }),
    );

    await screen.findByText("Could not start that task");
    expect(writes).toEqual([]);
    expect(screen.getByRole("button", { name: TASKS[1].title })).toHaveProperty(
      "disabled",
      false,
    );
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
        { name: TASKS[0].title },
        { timeout: 5000 },
      ),
    ).toBeDefined();
    expect(screen.queryByText(generic[0].title)).toBeNull();
  });
});

describe("Suggested first tasks — Home after the first message", () => {
  it("shows the rest of the list, and the started one as in progress", async () => {
    stored[0] = { ...stored[0], state: "sent", session_id: "session-1" };
    renderHome();

    const block = await findBlock();
    const started = within(block).getByRole("link", {
      name: /Chase unpaid Shopify invoices/,
    });
    expect(started.getAttribute("href")).toBe("/copilot?sessionId=session-1");
    expect(within(started).getByText("In progress")).toBeDefined();
    expect(
      within(block).getByRole("button", { name: TASKS[1].title }),
    ).toBeDefined();
    expect(
      within(block).getByRole("button", { name: TASKS[2].title }),
    ).toBeDefined();
  });

  it("hides a row, and Undo puts it back", async () => {
    renderHome();
    const block = await findBlock();

    await userEvent.click(
      within(block).getByRole("button", { name: `Hide: ${TASKS[1].title}` }),
    );

    await waitFor(() =>
      expect(
        within(block).queryByRole("button", { name: TASKS[1].title }),
      ).toBeNull(),
    );
    await waitFor(() =>
      expect(writes).toEqual([{ task_ids: ["1"], state: "dismissed" }]),
    );
    expect(events("home_suggestion_dismissed")).toEqual([
      { position: 1, surface: "home", remaining: 2, scope: "one" },
    ]);

    await screen.findByText("Suggestion hidden");
    await userEvent.click(screen.getByRole("button", { name: "Undo" }));

    expect(
      await within(block).findByRole("button", { name: TASKS[1].title }),
    ).toBeDefined();
    await waitFor(() =>
      expect(writes.at(-1)).toEqual({ task_ids: ["1"], state: "open" }),
    );
    expect(events("home_suggestion_dismissed")).toHaveLength(1);
  });

  it("stays hidden after a reload", async () => {
    stored[2] = { ...stored[2], state: "dismissed" };
    renderHome();

    const block = await findBlock();

    expect(within(block).getAllByTestId("suggested-task")).toHaveLength(2);
    expect(
      within(block).queryByRole("button", { name: TASKS[2].title }),
    ).toBeNull();
  });

  it("Hide all empties the block in one write, and Undo restores it", async () => {
    renderHome();
    const block = await findBlock();

    await userEvent.click(
      within(block).getByRole("button", { name: "Hide all" }),
    );

    await waitFor(() =>
      expect(
        screen.queryByRole("region", { name: "Suggested first tasks" }),
      ).toBeNull(),
    );
    await waitFor(() =>
      expect(writes).toEqual([
        { task_ids: ["0", "1", "2"], state: "dismissed" },
      ]),
    );
    expect(events("home_suggestion_dismissed")).toEqual([
      { position: null, surface: "home", remaining: 0, scope: "all" },
    ]);

    await screen.findByText("Suggestions hidden");
    await userEvent.click(screen.getByRole("button", { name: "Undo" }));

    const restored = await findBlock();
    expect(within(restored).getAllByTestId("suggested-task")).toHaveLength(3);
  });

  it("leaves the generic starters off Home for a user who never got a list", async () => {
    isStored = false;
    renderHome();

    await settleList();

    expect(
      screen.queryByRole("region", { name: "Suggested first tasks" }),
    ).toBeNull();
  });

  it("goes while an expert is the recipient", async () => {
    server.use(
      getListExpertIdentitiesMockHandler([
        {
          id: "expert-maria",
          name: "Maria",
          role: "Marketing Strategist",
          avatar_url: null,
          is_archived: false,
        },
      ]),
    );
    renderHome({ searchParams: "?expertId=expert-maria" });

    await screen.findByPlaceholderText("What should Maria work on?");
    await settleList();

    expect(
      screen.queryByRole("region", { name: "Suggested first tasks" }),
    ).toBeNull();
  });
});
