import { getListExpertIdentitiesMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { ExpertIdentity } from "@/app/api/__generated__/models/expertIdentity";
import { SidebarProvider } from "@/components/molecules/Sidebar/Sidebar";
import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { RecentChats } from "../RecentChats";
import { type SidebarChat, sidebarBackend } from "./sidebarBackend";

// SECRT-3401: a new expert had no sidebar entry until their first chat, and
// nothing pointed the user at that chat. Experts added in the last week with
// no chat get their own group, with one row that opens it.

const NOW = new Date("2026-11-09T12:00:00Z");

function expert(
  id: string,
  name: string,
  createdAt: string | null,
  overrides: Partial<ExpertIdentity> = {},
): ExpertIdentity {
  return {
    id,
    name,
    avatar_url: null,
    role: "Specialist",
    is_archived: false,
    created_at: createdAt ? new Date(createdAt) : null,
    ...overrides,
  };
}

function chat(id: string, expertId: string | null): SidebarChat {
  return { id, title: `${id} title`, expert_id: expertId };
}

function renderWith(experts: ExpertIdentity[], chats: SidebarChat[]) {
  server.use(
    ...sidebarBackend(() => chats),
    getListExpertIdentitiesMockHandler(experts),
  );
  return render(
    <SidebarProvider>
      <RecentChats />
    </SidebarProvider>,
  );
}

function isBefore(first: Element, second: Element) {
  return !!(
    first.compareDocumentPosition(second) & Node.DOCUMENT_POSITION_FOLLOWING
  );
}

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  vi.setSystemTime(NOW);
});

afterEach(() => {
  vi.useRealTimers();
  server.resetHandlers();
});

describe("a new expert in Recent chats", () => {
  it("is listed before their first chat, with a way to say hello", async () => {
    renderWith([expert("expert-nadia", "Nadia", "2026-11-09T11:00:00Z")], []);

    const hello = await screen.findByRole("link", {
      name: "Say hello to Nadia",
    });
    // The kickoff link: Nadia introduces herself in her first chat.
    expect(hello.getAttribute("href")).toBe(
      "/copilot?expertId=expert-nadia&kickoff=1",
    );
    expect(screen.queryByText("No conversations yet")).toBeNull();
  });

  it("is marked New beside their name", async () => {
    renderWith([expert("expert-nadia", "Nadia", "2026-11-09T11:00:00Z")], []);

    const group = await screen.findByRole("group", { name: "Nadia" });
    expect(within(group).getByText("Nadia")).toBeDefined();
    expect(within(group).getByText("New")).toBeDefined();
    expect(
      within(group).getByRole("link", { name: "Say hello to Nadia" }),
    ).toBeDefined();
  });

  it("lists the newest first, after Otto and before the experts already in use", async () => {
    renderWith(
      [
        expert("expert-maria", "Maria", "2026-10-01T09:00:00Z"),
        expert("expert-ana", "Ana", "2026-11-05T09:00:00Z"),
        expert("expert-nadia", "Nadia", "2026-11-09T11:00:00Z"),
      ],
      [chat("otto-1", null), chat("maria-1", "expert-maria")],
    );

    const hellos = await screen.findAllByRole("link", {
      name: /^Say hello to /,
    });
    expect(hellos.map((link) => link.textContent)).toEqual([
      "Say hello to Nadia",
      "Say hello to Ana",
    ]);
    const otto = screen.getByRole("button", { name: "Otto chats" });
    const maria = screen.getByRole("button", { name: "Maria chats" });
    expect(isBefore(otto, hellos[0])).toBe(true);
    expect(isBefore(hellos[1], maria)).toBe(true);
  });

  it("gives way to their normal group once a chat exists", async () => {
    renderWith(
      [expert("expert-nadia", "Nadia", "2026-11-09T11:00:00Z")],
      [chat("nadia-1", "expert-nadia")],
    );

    expect(
      await screen.findByRole("button", { name: "Nadia chats" }),
    ).toBeDefined();
    expect(screen.queryByRole("link", { name: /Say hello/ })).toBeNull();
    expect(screen.queryByText("New")).toBeNull();
  });

  it("leaves out experts added over a week ago, fired ones, and unknown join dates", async () => {
    renderWith(
      [
        expert("expert-old", "Olga", "2026-11-01T11:00:00Z"),
        expert("expert-fired", "Finn", "2026-11-09T09:00:00Z", {
          is_archived: true,
        }),
        expert("expert-undated", "Uma", null),
      ],
      [],
    );

    expect(await screen.findByText("No conversations yet")).toBeDefined();
    expect(screen.queryByRole("link", { name: /Say hello/ })).toBeNull();
  });
});
