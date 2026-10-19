import { SESSION_LIST_REFETCH_INTERVAL_MS } from "@/app/(platform)/copilot/useSessionList";
import { SidebarProvider } from "@/components/ui/sidebar";
import { server } from "@/mocks/mock-server";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import type { ComponentProps } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useCopilotUIStore } from "../../../store";
import { ChatSidebar } from "../ChatSidebar";

const blockRenders = vi.fn();

// Every render of a row renders its `ChatSessionBlock`, so counting the block
// counts the rows that re-rendered (and which ones).
vi.mock("../../ChatSessionBlock/ChatSessionBlock", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("../../ChatSessionBlock/ChatSessionBlock")
    >();
  function CountedChatSessionBlock(
    props: ComponentProps<typeof actual.ChatSessionBlock>,
  ) {
    blockRenders(props.title);
    return <actual.ChatSessionBlock {...props} />;
  }
  return { ...actual, ChatSessionBlock: CountedChatSessionBlock };
});

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) => flag === "chat-pinning",
  };
});

const exportChat = vi.fn();
vi.mock("../../../helpers/exportChatAsMarkdown", () => ({
  fetchAndExportChat: (id: string, title: string | null | undefined) =>
    exportChat(id, title),
}));

vi.mock("../../../sharing/ShareChatDialog", () => ({
  ShareChatDialog: ({ sessionId }: { sessionId: string }) => (
    <div role="dialog" aria-label={`Share ${sessionId}`} />
  ),
}));

vi.mock("../../UsageLimits/UsagePopover/UsagePopover", () => ({
  UsagePopover: () => null,
}));
vi.mock("../components/NotificationToggle/NotificationToggle", () => ({
  NotificationToggle: () => null,
}));

interface SessionSeed {
  id: string;
  title: string;
  is_pinned?: boolean;
  updated_at?: string;
}

function toSummary(seed: SessionSeed) {
  return {
    id: seed.id,
    title: seed.title,
    is_processing: false,
    is_pinned: seed.is_pinned ?? false,
    chat_status: "idle",
    // `Z`-suffixed, so the API client turns them into `Date`s as in the app.
    created_at: "2026-01-05T09:00:00Z",
    updated_at: seed.updated_at ?? "2026-01-05T09:00:00Z",
  };
}

let listed: SessionSeed[] = [];
let listRequests: string[] = [];

function serveList(total?: () => number) {
  server.use(
    http.get("*/api/chat/sessions", ({ request }) => {
      const url = new URL(request.url);
      const offset = Number(url.searchParams.get("offset") ?? "0");
      const limit = Number(url.searchParams.get("limit") ?? "50");
      listRequests.push(String(offset));
      return HttpResponse.json({
        sessions: listed.slice(offset, offset + limit).map(toSummary),
        total: total ? total() : listed.length,
      });
    }),
  );
}

function renderSidebar() {
  return render(
    <SidebarProvider>
      <ChatSidebar />
    </SidebarProvider>,
  );
}

async function tick(ms: number) {
  await act(async () => {
    vi.advanceTimersByTime(ms);
  });
}

/** Lets in-flight requests resolve and React commit what they returned. */
async function settle() {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 50));
  });
}

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  useCopilotUIStore.setState({
    sessionToDelete: null,
    completedSessionIDs: new Set(),
  });
  listed = [
    { id: "s1", title: "First chat" },
    { id: "s2", title: "Second chat" },
    { id: "s3", title: "Third chat" },
  ];
  listRequests = [];
  blockRenders.mockClear();
});

afterEach(() => {
  vi.useRealTimers();
  server.resetHandlers();
});

describe("ChatSidebar — session rows", () => {
  it("re-renders no row when a refetch returns the same sessions", async () => {
    serveList();
    renderSidebar();
    await screen.findByText("Third chat");
    await settle();
    blockRenders.mockClear();
    const requestsBefore = listRequests.length;

    await tick(SESSION_LIST_REFETCH_INTERVAL_MS);
    await waitFor(() =>
      expect(listRequests.length).toBeGreaterThan(requestsBefore),
    );
    await settle();

    expect(blockRenders).not.toHaveBeenCalled();
  });

  it("re-renders only the pinned row when one session is pinned", async () => {
    serveList();
    server.use(
      http.patch("*/api/chat/sessions/:id/pinned", async ({ params }) => {
        // The backend lists pinned chats first.
        const pinned = listed.find((s) => s.id === params.id);
        if (pinned) {
          listed = [
            { ...pinned, is_pinned: true },
            ...listed.filter((s) => s.id !== params.id),
          ];
        }
        return HttpResponse.json({ status: "ok" });
      }),
    );
    renderSidebar();
    const row = (await screen.findByText("Third chat")).closest("div.group");
    if (!row) throw new Error("row not found");
    await settle();
    blockRenders.mockClear();

    fireEvent.pointerDown(
      within(row as HTMLElement).getByRole("button", { name: /more actions/i }),
      { button: 0 },
    );
    fireEvent.click(await screen.findByRole("menuitem", { name: "Pin chat" }));

    await waitFor(() =>
      expect(blockRenders).toHaveBeenCalledWith("Third chat"),
    );
    await settle();
    expect(blockRenders.mock.calls.map(([title]) => title)).toEqual([
      "Third chat",
    ]);
  });
});

describe("ChatSidebar — session rows while editing and as they age", () => {
  it("does not re-render sibling rows while a title is being edited", async () => {
    serveList();
    renderSidebar();
    const row = (await screen.findByText("Third chat")).closest("div.group");
    if (!row) throw new Error("row not found");

    fireEvent.pointerDown(
      within(row as HTMLElement).getByRole("button", { name: /more actions/i }),
      { button: 0 },
    );
    fireEvent.click(await screen.findByRole("menuitem", { name: /rename/i }));
    const input = await screen.findByLabelText("Rename chat");
    await settle();
    blockRenders.mockClear();

    fireEvent.change(input, { target: { value: "Renamed" } });
    await settle();

    expect(blockRenders).not.toHaveBeenCalled();
  });

  it("re-renders a row when its 'Today' label turns into 'Yesterday'", async () => {
    // The rows were updated at 09:00 UTC; one becomes a day old first.
    listed = [
      { id: "s1", title: "First chat", updated_at: "2026-01-05T09:00:00Z" },
      { id: "s2", title: "Second chat", updated_at: "2026-01-05T15:00:00Z" },
    ];
    vi.setSystemTime(new Date("2026-01-06T08:59:00Z"));
    serveList();
    renderSidebar();
    await screen.findByText("Second chat");
    expect(screen.getAllByText("Today")).toHaveLength(2);
    await settle();
    blockRenders.mockClear();

    await tick(2 * 60_000);
    await settle();

    expect(blockRenders.mock.calls.map(([title]) => title)).toEqual([
      "First chat",
    ]);
    expect(screen.getByText("Yesterday")).toBeDefined();
    expect(screen.getAllByText("Today")).toHaveLength(1);
  });
});

describe("ChatSidebar — shared row callbacks", () => {
  async function openActionsFor(title: string) {
    const row = (await screen.findByText(title)).closest("div.group");
    if (!row) throw new Error(`row not found for ${title}`);
    fireEvent.pointerDown(
      within(row as HTMLElement).getByRole("button", { name: /more actions/i }),
      { button: 0 },
    );
  }

  it("runs each row action on that row's own session", async () => {
    const renames: string[] = [];
    serveList();
    server.use(
      http.patch("*/api/chat/sessions/:id/title", ({ params }) => {
        renames.push(String(params.id));
        return HttpResponse.json({ status: "ok" });
      }),
    );
    exportChat.mockClear();
    renderSidebar();

    await openActionsFor("Third chat");
    fireEvent.click(
      await screen.findByRole("menuitem", { name: /export chat/i }),
    );
    await waitFor(() =>
      expect(exportChat).toHaveBeenCalledWith("s3", "Third chat"),
    );

    await openActionsFor("Second chat");
    fireEvent.click(
      await screen.findByRole("menuitem", { name: /share chat/i }),
    );
    expect(
      await screen.findByRole("dialog", { name: "Share s2" }),
    ).toBeDefined();

    await openActionsFor("First chat");
    fireEvent.click(await screen.findByRole("menuitem", { name: /rename/i }));
    const input = await screen.findByLabelText("Rename chat");
    fireEvent.change(input, { target: { value: "Never saved" } });
    fireEvent.keyDown(input, { key: "Escape" });
    fireEvent.blur(input);
    await settle();

    expect(screen.queryByLabelText("Rename chat")).toBeNull();
    expect(screen.getByText("First chat")).toBeDefined();
    expect(renames).toEqual([]);
  });
});

describe("ChatSidebar — 10-second refresh", () => {
  function seedPages(count: number) {
    listed = Array.from({ length: count }, (_, i) => ({
      id: `s${i}`,
      title: `Chat ${i}`,
    }));
  }

  async function loadOlderChats() {
    fireEvent.click(
      await screen.findByRole("button", { name: /load older chats/i }),
    );
    await screen.findByText("Chat 74");
    await settle();
  }

  it("requests only the first page once older pages are loaded", async () => {
    seedPages(75);
    serveList();
    renderSidebar();
    await loadOlderChats();
    listRequests = [];

    await tick(SESSION_LIST_REFETCH_INTERVAL_MS);
    await waitFor(() => expect(listRequests.length).toBeGreaterThan(0));
    await settle();

    expect(listRequests).toEqual(["0"]);
  });

  it("still shows a renamed chat on the first page after one interval", async () => {
    seedPages(75);
    serveList();
    renderSidebar();
    await loadOlderChats();
    listRequests = [];

    listed[3] = { ...listed[3], title: "Renamed on the server" };
    await tick(SESSION_LIST_REFETCH_INTERVAL_MS);

    expect(await screen.findByText("Renamed on the server")).toBeDefined();
    expect(listRequests).toEqual(["0"]);
    expect(screen.getByText("Chat 74")).toBeDefined();
  });

  it("refetches every loaded page when a new chat shifts the first page", async () => {
    seedPages(75);
    serveList();
    renderSidebar();
    await loadOlderChats();

    listed = [{ id: "new", title: "Brand new chat" }, ...listed];
    await tick(SESSION_LIST_REFETCH_INTERVAL_MS);

    expect(await screen.findByText("Brand new chat")).toBeDefined();
    // Chat 49 was pushed from the first page onto the second; it must not
    // fall into the gap between them.
    await waitFor(() => expect(screen.getByText("Chat 49")).toBeDefined());
    for (let i = 0; i < 75; i++) {
      expect(screen.getAllByText(`Chat ${i}`)).toHaveLength(1);
    }
  });
});
