import { SESSION_LIST_REFETCH_INTERVAL_MS } from "@/app/(platform)/copilot/useSessionList";
import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { SidebarProvider } from "@/components/molecules/Sidebar/Sidebar";
import { server } from "@/mocks/mock-server";
import { Key, storage } from "@/services/storage/local-storage";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RecentChats } from "../RecentChats";
import {
  type SidebarChat,
  type SidebarRequest,
  groupChats,
  sidebarBackend,
} from "./sidebarBackend";

let chats: SidebarChat[] = [];
let requests: SidebarRequest[] = [];

function renderRecentChats() {
  return render(
    <SidebarProvider>
      <RecentChats />
    </SidebarProvider>,
  );
}

function reads(path: SidebarRequest["path"]) {
  return requests.filter((request) => request.path === path).length;
}

/** Lets in-flight requests resolve and React commit what they returned. */
async function settle() {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 50));
  });
}

async function nextRefresh() {
  const before = reads("session-groups");
  await act(async () => {
    vi.advanceTimersByTime(SESSION_LIST_REFETCH_INTERVAL_MS);
  });
  await waitFor(() => expect(reads("session-groups")).toBeGreaterThan(before));
  await settle();
}

async function showOlderOttoChats() {
  fireEvent.click(
    await screen.findByRole("button", { name: "Show more Otto chats" }),
  );
  await screen.findByText("autopilot chat 13");
  await settle();
}

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  vi.setSystemTime(new Date("2026-06-30T12:00:00Z"));
  useCopilotUIStore.setState({ sessionToDelete: null });
  storage.clean(Key.RECENT_CHATS_COLLAPSED_GROUPS);
  chats = groupChats(30);
  requests = [];
  server.use(...sidebarBackend(() => chats, requests));
});

afterEach(() => {
  vi.useRealTimers();
  server.resetHandlers();
});

describe("RecentChats — 10-second refresh", () => {
  it("shows a chat renamed on the server after one interval", async () => {
    renderRecentChats();
    await screen.findByText("autopilot chat 2");

    chats[1] = { ...chats[1], title: "Renamed on the server" };
    await nextRefresh();

    expect(screen.getByText("Renamed on the server")).toBeDefined();
  });

  it("refreshes the grouped read alone, not a group's older pages", async () => {
    renderRecentChats();
    await showOlderOttoChats();
    const pagesBefore = reads("sessions");

    await nextRefresh();

    expect(reads("sessions")).toBe(pagesBefore);
    expect(screen.getByText("autopilot chat 13")).toBeDefined();
  });

  it("puts a new chat on top of a group whose older chats are shown", async () => {
    renderRecentChats();
    await showOlderOttoChats();

    chats = [{ id: "new", title: "Brand new chat" }, ...chats];
    await nextRefresh();

    expect(screen.getByText("Brand new chat")).toBeDefined();
    for (let i = 1; i <= 12; i++) {
      expect(screen.getAllByText(`autopilot chat ${i}`)).toHaveLength(1);
    }
  });
});
