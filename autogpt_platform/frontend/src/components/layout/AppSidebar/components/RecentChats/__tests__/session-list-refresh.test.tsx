import { SESSION_LIST_REFETCH_INTERVAL_MS } from "@/app/(platform)/copilot/useSessionList";
import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { SidebarProvider } from "@/components/ui/sidebar";
import { server } from "@/mocks/mock-server";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RecentChats } from "../RecentChats";

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

interface SessionSeed {
  id: string;
  title: string;
}

let listed: SessionSeed[] = [];
let listRequests: string[] = [];

function toSummary(seed: SessionSeed) {
  return {
    id: seed.id,
    title: seed.title,
    is_processing: false,
    is_pinned: false,
    // `Z`-suffixed, so the API client turns them into `Date`s as in the app.
    created_at: "2026-06-30T10:00:00Z",
    updated_at: "2026-06-30T10:00:00Z",
  };
}

function seedPages(count: number) {
  listed = Array.from({ length: count }, (_, i) => ({
    id: `s${i}`,
    title: `Chat ${i}`,
  }));
}

function renderRecentChats() {
  return render(
    <SidebarProvider>
      <RecentChats />
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

async function loadOlderChats() {
  fireEvent.click(await screen.findByRole("button", { name: /load more/i }));
  fireEvent.click(
    await screen.findByRole("button", { name: "Load all Otto chats" }),
  );
  await screen.findByText("Chat 74");
  await settle();
}

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  vi.setSystemTime(new Date("2026-06-30T12:00:00Z"));
  useCopilotUIStore.setState({ sessionToDelete: null });
  seedPages(75);
  listRequests = [];
  server.use(
    http.get("*/api/chat/sessions", ({ request }) => {
      const url = new URL(request.url);
      const offset = Number(url.searchParams.get("offset") ?? "0");
      const limit = Number(url.searchParams.get("limit") ?? "50");
      listRequests.push(String(offset));
      return HttpResponse.json({
        sessions: listed.slice(offset, offset + limit).map(toSummary),
        total: listed.length,
      });
    }),
  );
});

afterEach(() => {
  vi.useRealTimers();
  server.resetHandlers();
});

describe("RecentChats — 10-second refresh", () => {
  it("requests only the first page once older pages are loaded", async () => {
    renderRecentChats();
    await loadOlderChats();
    listRequests = [];

    await tick(SESSION_LIST_REFETCH_INTERVAL_MS);
    await waitFor(() => expect(listRequests.length).toBeGreaterThan(0));
    await settle();

    expect(listRequests).toEqual(["0"]);
  });

  it("still shows a renamed chat on the first page after one interval", async () => {
    renderRecentChats();
    await loadOlderChats();
    listRequests = [];

    listed[3] = { ...listed[3], title: "Renamed on the server" };
    await tick(SESSION_LIST_REFETCH_INTERVAL_MS);

    expect(await screen.findByText("Renamed on the server")).toBeDefined();
    expect(listRequests).toEqual(["0"]);
    expect(screen.getByText("Chat 74")).toBeDefined();
  });

  it("refetches every loaded page when a new chat shifts the first page", async () => {
    renderRecentChats();
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
