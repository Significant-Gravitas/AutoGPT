import { getGetV2ListSessionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { getListExpertIdentitiesMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { SESSION_LIST_REFETCH_INTERVAL_MS } from "@/app/(platform)/copilot/useSessionList";
import { SidebarProvider } from "@/components/ui/sidebar";
import { server } from "@/mocks/mock-server";
import {
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
    useGetFlag: (flag: string) => flag === "hire-experts",
  };
});

const mariaExpert: Expert = {
  id: "expert-maria",
  name: "Maria",
  avatar_url: "https://example.com/maria.png",
  role: "Marketing Strategist",
  job_title: "Marketing Manager",
  bio: null,
  skills: [],
  tagline: "Grows your brand while you sleep",
  identity: "You are Maria, a senior marketing strategist.",
  voice_preferences: "Warm, concise, and direct.",
  boundaries: "Never invent customer evidence.",
  protected_soul_rules: [
    "The expert discloses that it is AI when acting externally.",
    "The expert asks for approval before acting externally.",
  ],
  is_template: false,
  source_template_id: "template-maria",
  is_archived: false,
  workflows: [],
};

function makeSession(args: {
  id: string;
  title: string;
  expertId?: string;
  isProcessing?: boolean;
}) {
  return {
    id: args.id,
    title: args.title,
    is_processing: args.isProcessing ?? false,
    created_at: "2026-06-30T10:00:00",
    updated_at: "2026-06-30T10:00:00",
    expert_id: args.expertId ?? null,
  };
}

function makeSessions(count: number, expertId?: string) {
  const prefix = expertId ?? "autopilot";
  return Array.from({ length: count }, (_, index) =>
    makeSession({
      id: `${prefix}-${index + 1}`,
      title: `${prefix} chat ${index + 1}`,
      expertId,
    }),
  );
}

function renderRecentChats() {
  return render(
    <SidebarProvider>
      <RecentChats />
    </SidebarProvider>,
  );
}

function groupHeader(label: string) {
  return screen.getByRole("button", { name: `${label} chats` });
}

/** Groups render open, so this is how a test gets to the collapsed state. */
async function collapseGroup(label: string) {
  fireEvent.click(
    await screen.findByRole("button", { name: `${label} chats` }),
  );
}

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
});

afterEach(() => {
  vi.useRealTimers();
  server.resetHandlers();
});

describe("RecentChats — expert groups", () => {
  it("starts open and collapses when its header is clicked", async () => {
    const sessions = makeSessions(2);
    server.use(
      getGetV2ListSessionsMockHandler200({ sessions, total: sessions.length }),
      getListExpertIdentitiesMockHandler([]),
    );
    renderRecentChats();

    expect(await screen.findByText("autopilot chat 1")).toBeDefined();

    fireEvent.click(groupHeader("Otto"));
    expect(screen.queryByText("autopilot chat 1")).toBeNull();

    fireEvent.click(groupHeader("Otto"));
    expect(await screen.findByText("autopilot chat 1")).toBeDefined();
  });

  it("keeps a running chat visible while the group is collapsed", async () => {
    const sessions = [
      makeSession({ id: "running", title: "running chat", isProcessing: true }),
      ...makeSessions(2),
    ];
    server.use(
      getGetV2ListSessionsMockHandler200({ sessions, total: sessions.length }),
      getListExpertIdentitiesMockHandler([]),
    );
    renderRecentChats();
    await collapseGroup("Otto");

    expect(await screen.findByText("running chat")).toBeDefined();
    expect(screen.queryByText("autopilot chat 1")).toBeNull();
  });

  it("shows four chats and reveals four more at a time", async () => {
    const sessions = makeSessions(10);
    server.use(
      getGetV2ListSessionsMockHandler200({ sessions, total: sessions.length }),
      getListExpertIdentitiesMockHandler([]),
    );
    renderRecentChats();

    expect(await screen.findByText("autopilot chat 4")).toBeDefined();
    expect(screen.queryByText("autopilot chat 5")).toBeNull();

    const loadMore = () =>
      screen.getByRole("button", { name: "Load more Otto chats" });

    fireEvent.click(loadMore());
    expect(await screen.findByText("autopilot chat 8")).toBeDefined();
    expect(screen.queryByText("autopilot chat 9")).toBeNull();

    fireEvent.click(loadMore());
    expect(await screen.findByText("autopilot chat 10")).toBeDefined();
    expect(
      screen.queryByRole("button", { name: "Load more Otto chats" }),
    ).toBeNull();
  });

  it("groups expert chats under the expert's name with independent previews", async () => {
    const sessions = [...makeSessions(12), ...makeSessions(11, mariaExpert.id)];
    server.use(
      getGetV2ListSessionsMockHandler200({ sessions, total: sessions.length }),
      getListExpertIdentitiesMockHandler([mariaExpert]),
    );
    renderRecentChats();

    expect(await screen.findByText("expert-maria chat 4")).toBeDefined();
    expect(await screen.findByText("Marketing Manager")).toBeDefined();
    expect(
      screen.getByText("Marketing Manager").classList.contains("opacity-70"),
    ).toBe(true);
    expect(screen.queryByText(mariaExpert.role)).toBeNull();
    expect(screen.queryByText("expert-maria chat 5")).toBeNull();
    expect(screen.queryByText("autopilot chat 5")).toBeNull();

    fireEvent.click(
      screen.getByRole("button", { name: "Load more Maria chats" }),
    );
    expect(await screen.findByText("expert-maria chat 8")).toBeDefined();
    expect(screen.queryByText("expert-maria chat 9")).toBeNull();
    expect(screen.queryByText("autopilot chat 5")).toBeNull();
  });

  it("falls back to a generic Expert label when the expert is unknown", async () => {
    const sessions = makeSessions(2, "expert-ghost");
    server.use(
      getGetV2ListSessionsMockHandler200({ sessions, total: sessions.length }),
      getListExpertIdentitiesMockHandler([]),
    );
    renderRecentChats();

    const expertGroup = await screen.findByRole("button", {
      name: "Expert chats",
    });
    expect(expertGroup.querySelector("img")).not.toBe(null);
    expect(await screen.findByText("expert-ghost chat 1")).toBeDefined();
  });

  it("uses a stable warm-stone fallback independent of the accent", async () => {
    const novaExpert: Expert = {
      ...mariaExpert,
      id: "expert-nova",
      name: "Nova",
      avatar_url: null,
      color: "violet-300",
    };
    const sessions = makeSessions(2, novaExpert.id);
    server.use(
      getGetV2ListSessionsMockHandler200({ sessions, total: sessions.length }),
      getListExpertIdentitiesMockHandler([novaExpert]),
    );
    renderRecentChats();

    const expertGroup = await screen.findByRole("button", {
      name: "Nova chats",
    });
    const avatar = expertGroup.querySelector("img");
    expect(avatar?.getAttribute("width")).toBe("32");
    expect(avatar?.getAttribute("height")).toBe("32");
    expect(avatar?.getAttribute("src")).toBe(
      "/autogpt-characters/v2.1/expert-general-01/neutral/32.webp",
    );
  });

  it("keeps the group-level and list-level Load more buttons distinct", async () => {
    const sessions = makeSessions(12);
    server.use(
      getGetV2ListSessionsMockHandler200({
        sessions,
        total: sessions.length + 10,
      }),
      getListExpertIdentitiesMockHandler([]),
    );
    renderRecentChats();

    expect(
      await screen.findByRole("button", { name: "Load more Otto chats" }),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Load more" })).toBeDefined();
  });

  it("keeps reveal and collapse state across session list refetches", async () => {
    const sessions = [...makeSessions(12), ...makeSessions(2, mariaExpert.id)];
    let listCalls = 0;
    server.use(
      http.get("*/api/chat/sessions", () => {
        listCalls++;
        return HttpResponse.json({ sessions, total: sessions.length });
      }),
      getListExpertIdentitiesMockHandler([mariaExpert]),
    );
    renderRecentChats();

    await collapseGroup("Maria");
    fireEvent.click(
      await screen.findByRole("button", { name: "Load more Otto chats" }),
    );
    expect(await screen.findByText("autopilot chat 8")).toBeDefined();

    const callsBefore = listCalls;
    vi.advanceTimersByTime(SESSION_LIST_REFETCH_INTERVAL_MS);
    await waitFor(() => expect(listCalls).toBeGreaterThan(callsBefore));

    expect(screen.getByText("autopilot chat 8")).toBeDefined();
    expect(screen.queryByText("expert-maria chat 1")).toBeNull();
  });

  it("links each group header to a new chat, except for fired experts", async () => {
    const maxExpert: Expert = {
      ...mariaExpert,
      id: "expert-max",
      name: "Max",
      is_archived: true,
    };
    const sessions = [
      ...makeSessions(1),
      ...makeSessions(1, mariaExpert.id),
      ...makeSessions(1, maxExpert.id),
    ];
    server.use(
      getGetV2ListSessionsMockHandler200({ sessions, total: sessions.length }),
      getListExpertIdentitiesMockHandler([mariaExpert, maxExpert]),
    );
    renderRecentChats();

    const mariaLink = await screen.findByRole("link", {
      name: "New chat with Maria",
    });
    // `new=1` is what stops the page adopting Maria's latest thread, which
    // is exactly the chat the + is meant to step out of.
    expect(mariaLink.getAttribute("href")).toBe(
      "/home?expertId=expert-maria&new=1",
    );
    expect(
      screen
        .getByRole("link", { name: "New chat with Otto" })
        .getAttribute("href"),
    ).toBe("/home");
    expect(groupHeader("Max")).toBeDefined();
    expect(
      screen.queryByRole("link", { name: "New chat with Max" }),
    ).toBeNull();
  });

  it("keeps the + link fresh even when the expert's latest chat is running", async () => {
    const sessions = [
      makeSession({
        id: "maria-running",
        title: "maria running chat",
        expertId: mariaExpert.id,
        isProcessing: true,
      }),
      ...makeSessions(1, mariaExpert.id),
    ];
    server.use(
      getGetV2ListSessionsMockHandler200({ sessions, total: sessions.length }),
      getListExpertIdentitiesMockHandler([mariaExpert]),
    );
    renderRecentChats();

    const mariaLink = await screen.findByRole("link", {
      name: "New chat with Maria",
    });
    const href = new URL(mariaLink.getAttribute("href") ?? "", "http://x");
    expect(href.pathname).toBe("/home");
    expect(href.searchParams.get("expertId")).toBe(mariaExpert.id);
    expect(href.searchParams.get("new")).toBe("1");
    expect(href.searchParams.has("sessionId")).toBe(false);
  });

  it("hides the group chevron at rest on desktop and reveals it on hover or focus", async () => {
    const sessions = [...makeSessions(1), ...makeSessions(1, mariaExpert.id)];
    server.use(
      getGetV2ListSessionsMockHandler200({ sessions, total: sessions.length }),
      getListExpertIdentitiesMockHandler([mariaExpert]),
    );
    renderRecentChats();

    const header = await screen.findByRole("button", { name: "Maria chats" });
    const chevron = [...header.querySelectorAll("svg")].at(-1);
    expect(chevron).toBeDefined();
    const classes = chevron!.classList;
    // Hidden at rest, but only above the touch breakpoint: a phone has no
    // hover state to reveal it with.
    expect(classes.contains("md:opacity-0")).toBe(true);
    expect(classes.contains("group-hover/expert-header:opacity-100")).toBe(
      true,
    );
    expect(
      classes.contains("group-focus-within/expert-header:opacity-100"),
    ).toBe(true);
    // Hidden via opacity, so the slot stays reserved and the open/closed
    // rotation still applies: no layout shift on hover.
    expect(classes.contains("size-5")).toBe(true);
    expect(
      classes.contains("group-data-[state=open]/expert-group:rotate-180"),
    ).toBe(true);
    expect(classes.contains("hidden")).toBe(false);

    // The Otto group has no reveal-on-hover + link of its own to lean on, so
    // its chevron must carry the same treatment.
    const ottoChevron = [...groupHeader("Otto").querySelectorAll("svg")].at(-1);
    expect(ottoChevron?.classList.contains("md:opacity-0")).toBe(true);
    expect(
      ottoChevron?.classList.contains("group-hover/expert-header:opacity-100"),
    ).toBe(true);
  });
});
