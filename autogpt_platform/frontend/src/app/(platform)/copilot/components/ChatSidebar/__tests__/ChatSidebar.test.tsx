import {
  getDeleteV2DeleteSessionMockHandler,
  getDeleteV2DeleteSessionMockHandler422,
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
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useGlobalSearchStore } from "@/app/(platform)/components/GlobalSearchModal/useGlobalSearchStore";
import { useCopilotUIStore } from "../../../store";
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
    useGetFlag: (flag: string) => flag === "chat-search",
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

  it("closes the dialog after a successful delete", async () => {
    server.use(getDeleteV2DeleteSessionMockHandler());
    renderSidebar();

    await openDeleteDialogFor("Other chat");
    expect(
      await screen.findByRole("heading", { name: /delete chat/i }),
    ).toBeDefined();
    const confirmButton = await screen.findByRole("button", {
      name: /^delete$/i,
    });
    fireEvent.click(confirmButton);

    await vi.waitFor(() => {
      expect(useCopilotUIStore.getState().sessionToDelete).toBeNull();
    });
    await vi.waitFor(() => {
      expect(
        screen.queryByRole("heading", { name: /delete chat/i }),
      ).toBeNull();
    });
    expect(toastMock).not.toHaveBeenCalled();
  });

  it("toasts on delete failure and clears the staged session", async () => {
    server.use(getDeleteV2DeleteSessionMockHandler422());
    renderSidebar();

    await openDeleteDialogFor("Active chat");
    const confirmButton = await screen.findByRole("button", {
      name: /^delete$/i,
    });
    fireEvent.click(confirmButton);

    await vi.waitFor(() => {
      expect(toastMock).toHaveBeenCalledTimes(1);
    });
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
  });
});
