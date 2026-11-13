import { SESSION_LIST_REFETCH_INTERVAL_MS } from "@/app/(platform)/copilot/useSessionList";
import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { SidebarProvider } from "@/components/molecules/Sidebar/Sidebar";
import { server } from "@/mocks/mock-server";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import type { ComponentProps } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RecentChats } from "../RecentChats";
import {
  type SidebarChat,
  type SidebarRequest,
  sidebarBackend,
} from "./sidebarBackend";

const itemRenders = vi.fn();

// Each chat row renders one `SidebarMenuItem`, so counting those counts the
// rows that re-rendered.
vi.mock("@/components/molecules/Sidebar/Sidebar", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Sidebar/Sidebar")
    >();
  function CountedSidebarMenuItem(
    props: ComponentProps<typeof actual.SidebarMenuItem>,
  ) {
    itemRenders();
    return <actual.SidebarMenuItem {...props} />;
  }
  return { ...actual, SidebarMenuItem: CountedSidebarMenuItem };
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

let listed: SidebarChat[] = [];
let requests: SidebarRequest[] = [];

function groupedReads() {
  return requests.filter((request) => request.path === "session-groups").length;
}

function renderRecentChats() {
  render(
    <SidebarProvider>
      <RecentChats />
    </SidebarProvider>,
  );
}

async function settle() {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 50));
  });
}

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  vi.setSystemTime(new Date("2026-06-30T12:00:00Z"));
  useCopilotUIStore.setState({ sessionToDelete: null });
  listed = [
    { id: "s1", title: "First chat" },
    { id: "s2", title: "Second chat" },
    { id: "s3", title: "Third chat" },
  ];
  requests = [];
  itemRenders.mockClear();
  server.use(
    ...sidebarBackend(() => listed, requests),
    // Pinning leaves a chat's place in time alone: the grouped read orders by
    // activity, and pinned chats leave their group for the Pinned section.
    http.patch("*/api/chat/sessions/:id/pinned", ({ params }) => {
      listed = listed.map((s) =>
        s.id === params.id ? { ...s, is_pinned: true } : s,
      );
      return HttpResponse.json({ status: "ok" });
    }),
  );
});

afterEach(() => {
  vi.useRealTimers();
  server.resetHandlers();
});

describe("RecentChats — chat rows", () => {
  it("re-renders no row when a refetch returns the same sessions", async () => {
    renderRecentChats();
    await screen.findByText("Third chat");
    await settle();
    itemRenders.mockClear();
    const readsBefore = groupedReads();

    await act(async () => {
      vi.advanceTimersByTime(SESSION_LIST_REFETCH_INTERVAL_MS);
    });
    await waitFor(() => expect(groupedReads()).toBeGreaterThan(readsBefore));
    await settle();

    expect(itemRenders).not.toHaveBeenCalled();
  });

  it("re-renders only the pinned row when one chat is pinned", async () => {
    renderRecentChats();
    await screen.findByText("Third chat");
    await settle();
    itemRenders.mockClear();

    const actions = screen.getAllByRole("button", { name: /chat actions/i });
    fireEvent.pointerDown(actions[2], { button: 0 });
    fireEvent.click(await screen.findByRole("menuitem", { name: "Pin chat" }));

    expect(await screen.findByText("Pinned")).toBeDefined();
    await settle();
    expect(itemRenders).toHaveBeenCalledTimes(1);
  });

  it("re-renders only the edited row while typing a rename", async () => {
    renderRecentChats();
    await screen.findByText("Third chat");
    await settle();

    const actions = screen.getAllByRole("button", { name: /chat actions/i });
    fireEvent.pointerDown(actions[2], { button: 0 });
    fireEvent.click(await screen.findByRole("menuitem", { name: "Rename" }));
    await screen.findByLabelText("Rename chat");
    await settle();
    itemRenders.mockClear();

    fireEvent.change(screen.getByLabelText("Rename chat"), {
      target: { value: "Renamed chat" },
    });
    await settle();

    expect(itemRenders).toHaveBeenCalledTimes(1);
  });
});
