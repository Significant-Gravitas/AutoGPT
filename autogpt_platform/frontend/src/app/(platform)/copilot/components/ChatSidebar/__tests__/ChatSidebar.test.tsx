import {
  getDeleteV2DeleteSessionMockHandler,
  getGetV2ListSessionsMockHandler200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";
import { SidebarProvider } from "@/components/ui/sidebar";
import { http, HttpResponse } from "msw";
import { parseAsString, useQueryState } from "nuqs";
import { withNuqsTestingAdapter } from "nuqs/adapters/testing";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useGlobalSearchStore } from "@/app/(platform)/components/GlobalSearchModal/useGlobalSearchStore";
import { useCopilotUIStore } from "../../../store";
import { useSendMessage } from "../../../useSendMessage";
import { ChatSidebar } from "../ChatSidebar";

const toastMock = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return {
    ...actual,
    toast: (...args: unknown[]) => toastMock(...args),
  };
});

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) =>
      flag === "chat-search" || flag === "chat-pinning",
  };
});

vi.mock("../../UsageLimits/UsageLimits", () => ({
  UsageLimits: () => null,
}));
vi.mock("../../UsageLimits/UsagePopover/UsagePopover", () => ({
  UsagePopover: () => null,
}));
vi.mock("../components/NotificationToggle/NotificationToggle", () => ({
  NotificationToggle: () => null,
}));

const sessions = [
  {
    id: "s1",
    title: "Active chat",
    is_processing: false,
    source_platform: "discord",
    created_at: "2025-01-01T00:00:00Z",
    updated_at: "2025-01-01T00:00:00Z",
  },
  {
    id: "s2",
    title: "Other chat",
    is_processing: false,
    created_at: "2025-01-01T00:00:00Z",
    updated_at: "2025-01-01T00:00:00Z",
  },
];

function renderSidebar() {
  return render(
    <SidebarProvider>
      <ChatSidebar />
    </SidebarProvider>,
  );
}

async function openDeleteDialogFor(title: string) {
  const sessionRow = (await screen.findByText(title)).closest("div.group");
  if (!sessionRow) throw new Error(`row not found for ${title}`);
  const moreButton = within(sessionRow as HTMLElement).getByRole("button", {
    name: /more actions/i,
  });
  fireEvent.pointerDown(moreButton, { button: 0 });
  const deleteItem = await screen.findByRole("menuitem", {
    name: /delete chat/i,
  });
  fireEvent.click(deleteItem);
}

describe("ChatSidebar — delete flow", () => {
  beforeEach(() => {
    toastMock.mockClear();
    useCopilotUIStore.setState({ sessionToDelete: null, isSearchOpen: false });
    server.use(
      getGetV2ListSessionsMockHandler200({ sessions, total: sessions.length }),
    );
  });

  afterEach(() => {
    server.resetHandlers();
  });

  it("opens the delete dialog with the chosen session title", async () => {
    renderSidebar();

    await openDeleteDialogFor("Other chat");

    expect(
      await screen.findByRole("heading", { name: /delete chat/i }),
    ).toBeDefined();
    expect(screen.getByText(/"Other chat"/)).toBeDefined();
    expect(useCopilotUIStore.getState().sessionToDelete).toMatchObject({
      id: "s2",
      title: "Other chat",
    });
  });

  it("shows a platform logo for chats from an external platform", async () => {
    renderSidebar();

    expect(await screen.findByAltText("Discord")).toBeDefined();
  });

  it("clears the staged session when Cancel is clicked", async () => {
    renderSidebar();

    await openDeleteDialogFor("Active chat");
    expect(useCopilotUIStore.getState().sessionToDelete).not.toBeNull();

    const cancelButton = await screen.findByRole("button", { name: /cancel/i });
    fireEvent.click(cancelButton);

    expect(useCopilotUIStore.getState().sessionToDelete).toBeNull();
  });

  it("removes the chat from the list as soon as Delete is confirmed", async () => {
    let deleted = false;
    let releaseDelete = () => {};
    const deleteHeld = new Promise<void>((resolve) => {
      releaseDelete = resolve;
    });
    server.use(
      getGetV2ListSessionsMockHandler200(() => {
        const remaining = deleted
          ? sessions.filter((session) => session.id !== "s2")
          : sessions;
        return { sessions: remaining, total: remaining.length };
      }),
      getDeleteV2DeleteSessionMockHandler(async () => {
        await deleteHeld;
        deleted = true;
      }),
    );
    renderSidebar();

    await openDeleteDialogFor("Other chat");
    const confirmButton = await screen.findByRole("button", {
      name: /^delete$/i,
    });
    fireEvent.click(confirmButton);

    await vi.waitFor(() => {
      expect(screen.queryByText("Other chat")).toBeNull();
    });
    expect(screen.queryByRole("heading", { name: /delete chat/i })).toBeNull();
    expect(useCopilotUIStore.getState().sessionToDelete).toBeNull();
    expect(screen.getByText("Active chat")).toBeDefined();

    releaseDelete();

    await vi.waitFor(() => {
      expect(deleted).toBe(true);
    });
    expect(screen.queryByText("Other chat")).toBeNull();
    expect(toastMock).not.toHaveBeenCalled();
  });

  it("keeps a chat out of the list while its delete is pending, even after the list refetches", async () => {
    let activeTitle = "Active chat";
    let releaseDelete = () => {};
    const deleteHeld = new Promise<void>((resolve) => {
      releaseDelete = resolve;
    });
    server.use(
      getGetV2ListSessionsMockHandler200(() => ({
        sessions: sessions.map((session) =>
          session.id === "s1" ? { ...session, title: activeTitle } : session,
        ),
        total: sessions.length,
      })),
      getDeleteV2DeleteSessionMockHandler(async () => {
        await deleteHeld;
      }),
    );
    renderSidebar();

    await openDeleteDialogFor("Other chat");
    fireEvent.click(await screen.findByRole("button", { name: /^delete$/i }));
    await vi.waitFor(() => {
      expect(screen.queryByText("Other chat")).toBeNull();
    });

    activeTitle = "Active chat, refetched";
    fireEvent.click(screen.getByText("Active chat"));

    expect(await screen.findByText("Active chat, refetched")).toBeDefined();
    expect(screen.queryByText("Other chat")).toBeNull();

    releaseDelete();
  });

  it("leaves the open chat as soon as its delete is confirmed, so nothing more is sent to it", async () => {
    let releaseDelete = () => {};
    const deleteHeld = new Promise<void>((resolve) => {
      releaseDelete = resolve;
    });
    let deleted = false;
    server.use(
      getDeleteV2DeleteSessionMockHandler(async () => {
        await deleteHeld;
        deleted = true;
      }),
    );
    const sent: (string | null)[] = [];
    const createSession = vi.fn();
    const ActiveChatWrapper = withNuqsTestingAdapter({
      searchParams: "?sessionId=s1",
      hasMemory: true,
    });
    function ActiveChatComposer() {
      const [sessionId] = useQueryState("sessionId", parseAsString);
      const { onSend } = useSendMessage({
        sessionId,
        createSession,
        isUserStoppingRef: { current: false },
        sendMessage: async () => {
          sent.push(sessionId);
        },
      });
      return (
        <>
          <span data-testid="open-session">{sessionId ?? "none"}</span>
          <button onClick={() => onSend("One more thing")}>Send</button>
        </>
      );
    }
    render(
      <ActiveChatWrapper>
        <SidebarProvider>
          <ChatSidebar />
        </SidebarProvider>
        <ActiveChatComposer />
      </ActiveChatWrapper>,
    );
    expect(screen.getByTestId("open-session").textContent).toBe("s1");

    await openDeleteDialogFor("Active chat");
    fireEvent.click(await screen.findByRole("button", { name: /^delete$/i }));

    await vi.waitFor(() => {
      expect(screen.getByTestId("open-session").textContent).toBe("none");
    });
    expect(deleted).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await vi.waitFor(() => expect(createSession).toHaveBeenCalled());
    expect(sent).not.toContain("s1");

    releaseDelete();
    await vi.waitFor(() => expect(deleted).toBe(true));
  });

  it("puts the chat back and toasts when the delete fails, without waiting for a refetch", async () => {
    let holdListRequests = false;
    let rejectDelete = () => {};
    const deleteHeld = new Promise<void>((resolve) => {
      rejectDelete = resolve;
    });
    server.use(
      getGetV2ListSessionsMockHandler200(async () => {
        if (holdListRequests) await new Promise(() => {});
        return { sessions, total: sessions.length };
      }),
      http.delete("/api/proxy/api/chat/sessions/:sessionId", async () => {
        await deleteHeld;
        return HttpResponse.json({ detail: "boom" }, { status: 500 });
      }),
    );
    renderSidebar();

    await openDeleteDialogFor("Other chat");
    holdListRequests = true;
    fireEvent.click(await screen.findByRole("button", { name: /^delete$/i }));

    await vi.waitFor(() => {
      expect(screen.queryByText("Other chat")).toBeNull();
    });
    expect(screen.queryByRole("heading", { name: /delete chat/i })).toBeNull();

    rejectDelete();

    expect(await screen.findByText("Other chat")).toBeDefined();
    expect(toastMock).toHaveBeenCalledTimes(1);
    expect(toastMock.mock.calls[0][0]).toMatchObject({
      title: "Failed to delete chat",
      variant: "destructive",
    });
    expect(useCopilotUIStore.getState().sessionToDelete).toBeNull();
  });
});

describe("ChatSidebar — global search trigger", () => {
  beforeEach(() => {
    useGlobalSearchStore.setState({ isOpen: false });
    server.use(
      getGetV2ListSessionsMockHandler200({
        sessions: [],
        total: 0,
      }),
    );
  });

  // The search palette itself is mounted globally in the platform layout
  // (see GlobalSearchOverlay) — here we only assert the sidebar button opens
  // the shared store that drives it.
  it("opens the global search palette from the search button", async () => {
    const user = userEvent.setup();
    renderSidebar();

    expect(useGlobalSearchStore.getState().isOpen).toBe(false);

    await user.click(
      await screen.findByRole("button", { name: /search chats/i }),
    );

    expect(useGlobalSearchStore.getState().isOpen).toBe(true);
  });
});

describe("ChatSidebar — chat_status indicators", () => {
  afterEach(() => {
    server.resetHandlers();
  });

  it("renders the green running dot on sessions with chat_status='running'", async () => {
    const runningSessions = [
      {
        id: "s1",
        title: "Running essay",
        is_processing: false,
        chat_status: "running",
        created_at: "2025-01-01T00:00:00Z",
        updated_at: "2025-01-01T00:00:00Z",
      },
    ];
    server.use(
      getGetV2ListSessionsMockHandler200({
        sessions: runningSessions,
        total: 1,
      }),
    );
    renderSidebar();
    await screen.findByText("Running essay");
    expect(screen.getByTestId("session-status-running")).toBeDefined();
    expect(screen.queryByTestId("session-status-queued")).toBeNull();
  });

  it("renders the purple hourglass on sessions with chat_status='queued'", async () => {
    const queuedSessions = [
      {
        id: "s1",
        title: "Robot haiku",
        is_processing: false,
        chat_status: "queued",
        created_at: "2025-01-01T00:00:00Z",
        updated_at: "2025-01-01T00:00:00Z",
      },
    ];
    server.use(
      getGetV2ListSessionsMockHandler200({
        sessions: queuedSessions,
        total: 1,
      }),
    );
    renderSidebar();
    await screen.findByText("Robot haiku");
    expect(screen.getByTestId("session-status-queued")).toBeDefined();
    expect(screen.queryByTestId("session-status-running")).toBeNull();
  });

  it("renders no status indicator for chat_status='idle' sessions", async () => {
    const idleSessions = [
      {
        id: "s1",
        title: "Old chat",
        is_processing: false,
        chat_status: "idle",
        created_at: "2025-01-01T00:00:00Z",
        updated_at: "2025-01-01T00:00:00Z",
      },
    ];
    server.use(
      getGetV2ListSessionsMockHandler200({
        sessions: idleSessions,
        total: 1,
      }),
    );
    renderSidebar();
    await screen.findByText("Old chat");
    expect(screen.queryByTestId("session-status-running")).toBeNull();
    expect(screen.queryByTestId("session-status-queued")).toBeNull();
  });
});

describe("ChatSidebar — pinning", () => {
  afterEach(() => {
    server.resetHandlers();
  });

  async function openMenuFor(title: string) {
    const sessionRow = (await screen.findByText(title)).closest("div.group");
    if (!sessionRow) throw new Error(`row not found for ${title}`);
    const moreButton = within(sessionRow as HTMLElement).getByRole("button", {
      name: /more actions/i,
    });
    fireEvent.pointerDown(moreButton, { button: 0 });
  }

  it("pins an unpinned session via the dropdown", async () => {
    let pinnedBody: { is_pinned?: boolean } | null = null;
    server.use(
      getGetV2ListSessionsMockHandler200({
        sessions: [
          {
            id: "s1",
            title: "Unpinned chat",
            is_processing: false,
            is_pinned: false,
            created_at: "2025-01-01T00:00:00Z",
            updated_at: "2025-01-01T00:00:00Z",
          },
        ],
        total: 1,
      }),
      http.patch("*/api/chat/sessions/:id/pinned", async ({ request }) => {
        pinnedBody = (await request.json()) as { is_pinned: boolean };
        return HttpResponse.json({ status: "ok" });
      }),
    );
    renderSidebar();

    await openMenuFor("Unpinned chat");
    const pinItem = await screen.findByRole("menuitem", { name: /pin chat/i });
    fireEvent.click(pinItem);

    await vi.waitFor(() => {
      expect(pinnedBody).toEqual({ is_pinned: true });
    });
  });

  it("offers Unpin and a pinned indicator for pinned sessions", async () => {
    server.use(
      getGetV2ListSessionsMockHandler200({
        sessions: [
          {
            id: "s1",
            title: "Pinned chat",
            is_processing: false,
            is_pinned: true,
            created_at: "2025-01-01T00:00:00Z",
            updated_at: "2025-01-01T00:00:00Z",
          },
        ],
        total: 1,
      }),
    );
    renderSidebar();

    await screen.findByText("Pinned chat");
    expect(screen.getByTestId("session-pinned-indicator")).toBeDefined();

    await openMenuFor("Pinned chat");
    expect(
      await screen.findByRole("menuitem", { name: /unpin chat/i }),
    ).toBeDefined();
  });
});

describe("ChatSidebar — pagination", () => {
  afterEach(() => {
    server.resetHandlers();
  });

  function makeSessions(count: number, offset = 0) {
    return Array.from({ length: count }, (_, i) => ({
      id: `s${offset + i}`,
      title: `Chat ${offset + i}`,
      is_processing: false,
      created_at: "2025-01-01T00:00:00Z",
      updated_at: "2025-01-01T00:00:00Z",
    }));
  }

  it("hides 'Load older chats' when all sessions are loaded", async () => {
    server.use(
      getGetV2ListSessionsMockHandler200({
        sessions: makeSessions(3),
        total: 3,
      }),
    );
    renderSidebar();
    await screen.findByText("Chat 0");
    expect(
      screen.queryByRole("button", { name: /load older chats/i }),
    ).toBeNull();
  });

  it("renders 'Load older chats' when total exceeds loaded sessions", async () => {
    server.use(
      getGetV2ListSessionsMockHandler200({
        sessions: makeSessions(50),
        total: 75,
      }),
    );
    renderSidebar();
    await screen.findByText("Chat 0");
    expect(
      await screen.findByRole("button", { name: /load older chats/i }),
    ).toBeDefined();
  });

  it("fetches the next page with the loaded count as offset", async () => {
    const seenOffsets: string[] = [];
    server.use(
      http.get("*/api/chat/sessions", ({ request }) => {
        const offset = new URL(request.url).searchParams.get("offset") ?? "0";
        seenOffsets.push(offset);
        const offsetN = Number(offset);
        const sessions =
          offsetN === 0 ? makeSessions(50, 0) : makeSessions(25, 50);
        return HttpResponse.json({ sessions, total: 75 });
      }),
    );
    renderSidebar();

    const loadMore = await screen.findByRole("button", {
      name: /load older chats/i,
    });
    fireEvent.click(loadMore);

    await screen.findByText("Chat 50");
    expect(seenOffsets).toContain("50");
    await vi.waitFor(() => {
      expect(
        screen.queryByRole("button", { name: /load older chats/i }),
      ).toBeNull();
    });
  }, 10_000);
});

describe("ChatSidebar — rename", () => {
  afterEach(() => {
    server.resetHandlers();
  });

  async function startRenaming(title: string) {
    const sessionRow = (await screen.findByText(title)).closest("div.group");
    if (!sessionRow) throw new Error(`row not found for ${title}`);
    const moreButton = within(sessionRow as HTMLElement).getByRole("button", {
      name: /more actions/i,
    });
    fireEvent.pointerDown(moreButton, { button: 0 });
    fireEvent.click(await screen.findByRole("menuitem", { name: /rename/i }));
    return screen.findByLabelText("Rename chat");
  }

  it("keeps editing on Enter while an IME is composing, then commits on a real Enter", async () => {
    const titleBodies: { title: string }[] = [];
    server.use(
      getGetV2ListSessionsMockHandler200({ sessions, total: sessions.length }),
      http.patch("*/api/chat/sessions/:id/title", async ({ request }) => {
        titleBodies.push((await request.json()) as { title: string });
        return HttpResponse.json({ status: "ok" });
      }),
    );
    renderSidebar();

    const input = await startRenaming("Active chat");
    fireEvent.change(input, { target: { value: "新しいチャット" } });

    fireEvent.keyDown(input, { key: "Enter", isComposing: true });
    // The IME owns this Enter: the field stays in edit mode and, more to the
    // point, no rename request goes out.
    expect(screen.queryByLabelText("Rename chat")).not.toBeNull();
    expect(titleBodies).toEqual([]);

    fireEvent.keyDown(input, { key: "Enter" });
    await vi.waitFor(() => {
      expect(titleBodies).toEqual([{ title: "新しいチャット" }]);
    });
  });
});
