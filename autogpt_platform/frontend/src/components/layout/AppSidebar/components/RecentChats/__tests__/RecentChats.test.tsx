import { getListExpertIdentitiesMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { server } from "@/mocks/mock-server";
import { Key, storage } from "@/services/storage/local-storage";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { SidebarProvider } from "@/components/molecules/Sidebar/Sidebar";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { RecentChats } from "../RecentChats";
import {
  type SidebarChat,
  type SidebarRequest,
  groupChats,
  groupedRead,
  sidebarBackend,
} from "./sidebarBackend";

const mockEnabledFlags = new Set<string>();

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) => mockEnabledFlags.has(flag),
  };
});

let chats: SidebarChat[] = [];
let requests: SidebarRequest[] = [];

function serve(nextChats: SidebarChat[]) {
  chats = nextChats;
  server.use(...sidebarBackend(() => chats, requests));
}

function renderRecentChats() {
  return render(
    <SidebarProvider>
      <RecentChats />
    </SidebarProvider>,
  );
}

function ottoGroup() {
  const group = screen
    .getByRole("button", { name: "Otto chats" })
    .closest("div")?.parentElement;
  if (!group) throw new Error("Otto group not found");
  return group;
}

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  vi.setSystemTime(new Date("2026-06-30T12:00:00"));
  useCopilotUIStore.setState({ sessionToDelete: null });
  requests = [];
  storage.clean(Key.RECENT_CHATS_COLLAPSED_GROUPS);
  server.use(getListExpertIdentitiesMockHandler([]));
});

afterEach(() => {
  vi.useRealTimers();
  server.resetHandlers();
  mockEnabledFlags.delete("chat-pinning");
});

describe("RecentChats — empty + loading", () => {
  it("shows the empty state when there are no sessions", async () => {
    serve([]);
    renderRecentChats();
    expect(await screen.findByText(/no conversations yet/i)).toBeDefined();
  });

  it("says the chats failed to load, not that there are none, and retries", async () => {
    serve(groupChats(2));
    server.use(
      http.get(
        "*/api/chat/session-groups",
        () => HttpResponse.json({ detail: "Unavailable" }, { status: 503 }),
        { once: true },
      ),
    );
    renderRecentChats();

    const retry = await screen.findByRole("button", { name: /try again/i });
    expect(screen.queryByText(/no conversations yet/i)).toBeNull();

    fireEvent.click(retry);
    expect(await screen.findByText("autopilot chat 1")).toBeDefined();
  });
});

describe("RecentChats — grouped list", () => {
  it("shows Otto's chats from different dates together, without a click", async () => {
    serve([
      { id: "a", title: "Today chat" },
      { id: "b", title: "Yesterday chat" },
    ]);
    renderRecentChats();

    expect(await screen.findByText("Today chat")).toBeDefined();
    expect(screen.getByText("Yesterday chat")).toBeDefined();
    expect(screen.getByRole("button", { name: "Otto chats" })).toBeDefined();
    expect(screen.queryByText("Today")).toBeNull();
    expect(screen.queryByText("Yesterday")).toBeNull();
  });

  it("asks for three chats per group and 30 in all, in one read", async () => {
    serve(groupChats(2));
    renderRecentChats();
    await screen.findByText("autopilot chat 1");

    const reads = requests.filter((r) => r.path === "session-groups");
    expect(reads.length).toBeGreaterThan(0);
    expect(reads[0].params.get("min_per_group")).toBe("3");
    expect(reads[0].params.get("limit")).toBe("30");
    expect(requests.some((r) => r.path === "sessions")).toBe(false);
  });
});

describe("RecentChats — pinned split", () => {
  const pinnedAndLoose: SidebarChat[] = [
    { id: "loose", title: "Loose chat" },
    { id: "pinned", title: "Pinned chat", is_pinned: true },
  ];

  it("lifts pinned sessions out of the expert groups", async () => {
    mockEnabledFlags.add("chat-pinning");
    serve(pinnedAndLoose);
    renderRecentChats();

    const pinnedGroup = (await screen.findByText("Pinned")).closest(
      "div",
    )?.parentElement;
    if (!pinnedGroup) throw new Error("pinned group not found");
    expect(within(pinnedGroup).getByText("Pinned chat")).toBeDefined();
    expect(within(ottoGroup()).getByText("Loose chat")).toBeDefined();
    expect(within(ottoGroup()).queryByText("Pinned chat")).toBeNull();
    expect(
      within(screen.getByRole("button", { name: "Otto chats" })).getByText("1"),
    ).toBeDefined();
    const read = requests.find((r) => r.path === "session-groups");
    expect(read?.params.get("pinned_separately")).toBe("true");
  });

  it("does not greet a new expert as chat-less when their only chat is pinned", async () => {
    mockEnabledFlags.add("chat-pinning");
    server.use(
      getListExpertIdentitiesMockHandler([
        {
          id: "expert-nadia",
          name: "Nadia",
          avatar_url: null,
          role: "Researcher",
          is_archived: false,
          created_at: new Date("2026-06-30T11:00:00"),
        },
        {
          id: "expert-maria",
          name: "Maria",
          avatar_url: null,
          role: "Marketing",
          is_archived: false,
          created_at: new Date("2026-01-01T00:00:00"),
        },
      ]),
    );
    serve([
      {
        id: "nadia-1",
        title: "Nadia's pinned chat",
        expert_id: "expert-nadia",
        is_pinned: true,
      },
      { id: "maria-1", title: "Maria chat", expert_id: "expert-maria" },
    ]);
    renderRecentChats();

    expect(await screen.findByText("Nadia's pinned chat")).toBeDefined();
    // Maria's header is named from the expert list, so it is loaded by now.
    expect(
      await screen.findByRole("button", { name: "Maria chats" }),
    ).toBeDefined();
    expect(
      screen.queryByRole("link", { name: "Say hello to Nadia" }),
    ).toBeNull();
  });

  it("keeps pinned sessions in their expert group when the pinning flag is off", async () => {
    serve(pinnedAndLoose);
    renderRecentChats();

    await screen.findByText("Pinned chat");
    expect(screen.queryByText("Pinned")).toBeNull();
    expect(within(ottoGroup()).getByText("Pinned chat")).toBeDefined();
  });
});

describe("RecentChats — show more", () => {
  it("shows a small group whole, with no Show more", async () => {
    serve([{ id: "a", title: "Only" }]);
    renderRecentChats();
    await screen.findByText("Only");
    expect(screen.queryByRole("button", { name: /show more/i })).toBeNull();
  });

  it("has no list-wide Load more, only each group's own Show more", async () => {
    serve(groupChats(60));
    renderRecentChats();

    expect(
      await screen.findByRole("button", { name: "Show more Otto chats" }),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: /load more/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /load all/i })).toBeNull();
  });

  it("keeps Show more there to retry a read of older chats that failed", async () => {
    serve(groupChats(35));
    let failures = 0;
    server.use(
      http.get(
        "*/api/chat/sessions",
        () => {
          failures += 1;
          return HttpResponse.json({ detail: "Unavailable" }, { status: 503 });
        },
        { once: true },
      ),
    );
    renderRecentChats();

    fireEvent.click(
      await screen.findByRole("button", { name: "Show more Otto chats" }),
    );
    const retry = await screen.findByRole("button", {
      name: "Couldn't load more Otto chats. Try again",
    });
    expect(failures).toBe(1);
    expect(screen.queryByText("autopilot chat 31")).toBeNull();

    fireEvent.click(retry);
    expect(await screen.findByText("autopilot chat 35")).toBeDefined();
    expect(
      screen.queryByRole("button", { name: "Show more Otto chats" }),
    ).toBeNull();
  });

  it("asks only for the group's unpinned chats when pinned ones sit apart", async () => {
    mockEnabledFlags.add("chat-pinning");
    serve(groupChats(35));
    renderRecentChats();

    fireEvent.click(
      await screen.findByRole("button", { name: "Show more Otto chats" }),
    );
    expect(await screen.findByText("autopilot chat 35")).toBeDefined();
    const page = requests.find((r) => r.path === "sessions");
    expect(page?.params.get("pinned")).toBe("false");
  });
});

describe("RecentChats — delete flow", () => {
  it("stages a session for deletion when Delete is chosen", async () => {
    serve([{ id: "x", title: "Deletable" }]);
    renderRecentChats();

    const row = (await screen.findByText("Deletable")).closest("li");
    if (!row) throw new Error("row not found");
    fireEvent.pointerDown(
      within(row).getByRole("button", { name: /chat actions/i }),
      { button: 0 },
    );
    fireEvent.click(
      await screen.findByRole("menuitem", { name: /delete chat/i }),
    );

    expect(useCopilotUIStore.getState().sessionToDelete).toMatchObject({
      id: "x",
    });
  });

  it("keeps a deleted chat out of its group and count while the refetch is out", async () => {
    serve(groupChats(2));
    let deletes = 0;
    server.use(
      http.delete("*/api/chat/sessions/:id", ({ params }) => {
        deletes += 1;
        chats = chats.filter((chat) => chat.id !== params.id);
        return new HttpResponse(null, { status: 204 });
      }),
    );
    renderRecentChats();
    const row = (await screen.findByText("autopilot chat 1")).closest("li");
    if (!row) throw new Error("row not found");

    // Every grouped read after this point waits, so what the sidebar shows
    // once the delete has succeeded comes from its own cache.
    let release = () => {};
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    let heldReads = 0;
    let answeredReads = 0;
    server.use(
      http.get("*/api/chat/session-groups", async ({ request }) => {
        heldReads += 1;
        await held;
        answeredReads += 1;
        return HttpResponse.json(
          groupedRead(chats, new URL(request.url).searchParams),
        );
      }),
    );
    try {
      fireEvent.pointerDown(
        within(row).getByRole("button", { name: /chat actions/i }),
        { button: 0 },
      );
      fireEvent.click(
        await screen.findByRole("menuitem", { name: /delete chat/i }),
      );
      fireEvent.click(await screen.findByRole("button", { name: "Delete" }));
      // The refetch after the delete starts once the delete has settled, so
      // from here the chat is no longer hidden as "being deleted".
      await waitFor(() => expect(heldReads).toBeGreaterThan(0));
      expect(deletes).toBe(1);

      await waitFor(() => {
        expect(screen.queryByText("autopilot chat 1")).toBeNull();
        expect(
          within(screen.getByRole("button", { name: "Otto chats" })).getByText(
            "1",
          ),
        ).toBeDefined();
      });
      expect(screen.getByText("autopilot chat 2")).toBeDefined();
    } finally {
      release();
      await waitFor(() => expect(answeredReads).toBe(heldReads));
    }
  });
});
