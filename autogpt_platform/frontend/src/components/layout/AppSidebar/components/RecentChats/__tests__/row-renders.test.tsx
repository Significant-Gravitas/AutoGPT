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
import type { ComponentProps } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RecentChats } from "../RecentChats";

const itemRenders = vi.fn();

// Each chat row renders one `SidebarMenuItem`, so counting those counts the
// rows that re-rendered.
vi.mock("@/components/ui/sidebar", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/components/ui/sidebar")>();
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

interface SessionSeed {
  id: string;
  title: string;
  is_pinned?: boolean;
}

let listed: SessionSeed[] = [];
let listRequests = 0;

function toSummary(seed: SessionSeed) {
  return {
    id: seed.id,
    title: seed.title,
    is_processing: false,
    is_pinned: seed.is_pinned ?? false,
    // `Z`-suffixed, so the API client turns them into `Date`s as in the app.
    created_at: "2026-06-30T10:00:00Z",
    updated_at: "2026-06-30T10:00:00Z",
  };
}

async function renderRecentChats() {
  render(
    <SidebarProvider>
      <RecentChats />
    </SidebarProvider>,
  );
  fireEvent.click(
    await screen.findByRole("button", { name: "Load all Otto chats" }),
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
  listRequests = 0;
  itemRenders.mockClear();
  server.use(
    http.get("*/api/chat/sessions", () => {
      listRequests += 1;
      return HttpResponse.json({
        sessions: listed.map(toSummary),
        total: listed.length,
      });
    }),
    http.patch("*/api/chat/sessions/:id/pinned", ({ params }) => {
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
});

afterEach(() => {
  vi.useRealTimers();
  server.resetHandlers();
});

describe("RecentChats — chat rows", () => {
  it("re-renders no row when a refetch returns the same sessions", async () => {
    await renderRecentChats();
    await screen.findByText("Third chat");
    await settle();
    itemRenders.mockClear();
    const requestsBefore = listRequests;

    await act(async () => {
      vi.advanceTimersByTime(SESSION_LIST_REFETCH_INTERVAL_MS);
    });
    await waitFor(() => expect(listRequests).toBeGreaterThan(requestsBefore));
    await settle();

    expect(itemRenders).not.toHaveBeenCalled();
  });

  it("re-renders only the pinned row when one chat is pinned", async () => {
    await renderRecentChats();
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
    await renderRecentChats();
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
