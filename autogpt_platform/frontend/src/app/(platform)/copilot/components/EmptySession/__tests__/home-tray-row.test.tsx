import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { makeDashboard } from "@/app/(platform)/home/__tests__/heldItems";
import { getGetV2ListSessionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import {
  getListExpertIdentitiesMockHandler,
  getListExpertsMockHandler200,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { SuggestedTask } from "@/app/api/__generated__/models/suggestedTask";
import { server } from "@/mocks/mock-server";
import { Key, storage } from "@/services/storage/local-storage";
import {
  act,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { withNuqsTestingAdapter } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { EmptySession } from "../EmptySession";
import { makeSession } from "./taskFixtures";

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

const SUGGESTIONS_URL = "*/api/onboarding/brain-dump/suggested-tasks";

const TASKS: SuggestedTask[] = [
  {
    id: "0",
    title: "Chase unpaid Shopify invoices",
    prompt: "Chase every unpaid Shopify invoice with a polite follow-up.",
    state: "open",
    why: "Three invoices are more than 30 days late.",
    expert_id: null,
  },
  {
    id: "1",
    title: "Watch competitor pricing",
    prompt: "Check competitor pricing weekly and tell me what moved.",
    state: "open",
    why: null,
    expert_id: null,
  },
];

let isStored = true;
let reads = 0;

function withRecentChats() {
  server.use(
    getGetV2ListSessionsMockHandler200({
      sessions: [makeSession("session-a", "Weekly report", null)],
      total: 1,
    }),
  );
}

function renderHome(onStartTask = vi.fn(async () => "session-1")) {
  const Wrapper = withNuqsTestingAdapter({ searchParams: "", hasMemory: true });
  return render(
    <Wrapper>
      <EmptySession
        isCreatingSession={false}
        onCreateSession={() => {}}
        onSend={vi.fn()}
        onStartTask={onStartTask}
      />
    </Wrapper>,
  );
}

/** Lets the suggestions answer render, so an absence is not just "not loaded yet". */
async function settleSuggestions() {
  await waitFor(() => expect(reads).toBeGreaterThan(0));
  await act(async () => {});
}

function row() {
  return screen.getByTestId("home-tray-row");
}

function isBefore(first: Element, second: Element) {
  return Boolean(
    first.compareDocumentPosition(second) & Node.DOCUMENT_POSITION_FOLLOWING,
  );
}

beforeEach(() => {
  window.localStorage.clear();
  isStored = true;
  reads = 0;
  server.use(
    http.get(SUGGESTIONS_URL, () => {
      reads += 1;
      return HttpResponse.json({ tasks: TASKS, stored: isStored });
    }),
    getListExpertIdentitiesMockHandler([]),
    getListExpertsMockHandler200([]),
    http.get(/\/api\/proxy\/api\/home(?:\?.*)?$/, () =>
      HttpResponse.json(makeDashboard([])),
    ),
    getGetV2ListSessionsMockHandler200({ sessions: [], total: 0 }),
  );
});

describe("Home — suggested tasks beside recent chats", () => {
  it("puts the suggested tasks and the recent chats in one row under the composer, tasks first", async () => {
    withRecentChats();
    renderHome();

    const prompt = await screen.findByPlaceholderText(/What's your role/);
    const tasks = await screen.findByRole("region", {
      name: "Suggested first tasks",
    });
    const recent = await screen.findByRole("region", { name: "Recent chats" });

    expect(row().contains(tasks)).toBe(true);
    expect(row().contains(recent)).toBe(true);
    expect(row().contains(prompt)).toBe(false);
    expect(isBefore(prompt, row())).toBe(true);
    expect(isBefore(tasks, recent)).toBe(true);
    expect(
      within(tasks).getByRole("heading", { name: "Suggested first tasks" }),
    ).toBeDefined();
    expect(within(tasks).getAllByTestId("suggested-task")).toHaveLength(2);
  });

  it("shows the recent chats alone in the row when there are no suggested tasks", async () => {
    isStored = false;
    withRecentChats();
    renderHome();

    const recent = await screen.findByRole("region", { name: "Recent chats" });
    await settleSuggestions();

    expect(row().contains(recent)).toBe(true);
    expect(within(row()).getAllByRole("region")).toEqual([recent]);
    expect(
      screen.queryByRole("region", { name: "Suggested first tasks" }),
    ).toBeNull();
  });

  it("shows the suggested tasks alone in the row when there are no recent chats", async () => {
    renderHome();

    const tasks = await screen.findByRole("region", {
      name: "Suggested first tasks",
    });

    expect(within(row()).getAllByRole("region")).toEqual([tasks]);
    expect(screen.queryByRole("region", { name: "Recent chats" })).toBeNull();
  });

  it("leaves the row out when there is neither", async () => {
    isStored = false;
    renderHome();

    await screen.findByPlaceholderText(/What's your role/);
    await settleSuggestions();

    expect(screen.queryByTestId("home-tray-row")).toBeNull();
  });

  it("lists every task with no Hide toggle, even when the greeting's list was folded away", async () => {
    storage.set(Key.HOME_SUGGESTED_TASKS_COLLAPSED, JSON.stringify(["user-1"]));
    renderHome();

    const tasks = await screen.findByRole("region", {
      name: "Suggested first tasks",
    });

    expect(within(tasks).getAllByTestId("suggested-task")).toHaveLength(2);
    expect(
      within(tasks).queryByRole("button", { name: /Hide|Show/ }),
    ).toBeNull();
    expect(within(tasks).queryByText(/more task/)).toBeNull();
    expect(within(tasks).queryByText("Start here")).toBeNull();
  });

  it("starts a task with its owner when its row is clicked", async () => {
    const onStartTask = vi.fn(async () => "session-1");
    renderHome(onStartTask);

    await userEvent.click(
      await screen.findByRole("button", {
        name: "Start with Otto: Watch competitor pricing",
      }),
    );

    await waitFor(() =>
      expect(onStartTask).toHaveBeenCalledWith(
        "Check competitor pricing weekly and tell me what moved.",
      ),
    );
  });

  it("has no home recap under the row", async () => {
    renderHome();

    await screen.findByRole("region", { name: "Suggested first tasks" });

    expect(screen.queryByRole("heading", { name: "Your recap" })).toBeNull();
    expect(screen.queryByLabelText("Loading Home…")).toBeNull();
  });
});
