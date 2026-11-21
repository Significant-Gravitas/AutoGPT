import { getListExpertIdentitiesMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { SESSION_LIST_REFETCH_INTERVAL_MS } from "@/app/(platform)/copilot/useSessionList";
import { SidebarProvider } from "@/components/molecules/Sidebar/Sidebar";
import { server } from "@/mocks/mock-server";
import { Key, storage } from "@/services/storage/local-storage";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { RecentChats } from "../RecentChats";
import {
  type SidebarChat,
  type SidebarRequest,
  groupChats,
  sidebarBackend,
} from "./sidebarBackend";

// SECRT-3486: every group used to start collapsed except Otto's, which showed
// one chat, and the counts covered only the newest 50 chats overall. Now each
// group opens on its newest three, carries the server's real total, and pages
// in its own older chats.

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
    "The expert does the work you give it end to end, and gets your approval before acting externally on anything you did not ask for, unless you turn approvals off.",
  ],
  is_template: false,
  source_template_id: "template-maria",
  is_archived: false,
  workflows: [],
};

let chats: SidebarChat[] = [];
let requests: SidebarRequest[] = [];

function serve(nextChats: SidebarChat[], experts: Expert[] = [mariaExpert]) {
  chats = nextChats;
  server.use(
    ...sidebarBackend(() => chats, requests),
    getListExpertIdentitiesMockHandler(experts),
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

async function showMore(label: string) {
  fireEvent.click(
    await screen.findByRole("button", { name: `Show more ${label} chats` }),
  );
}

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  requests = [];
  storage.clean(Key.RECENT_CHATS_COLLAPSED_GROUPS);
});

afterEach(() => {
  vi.useRealTimers();
  server.resetHandlers();
});

describe("RecentChats — expert groups", () => {
  it("opens every group on its newest chats and toggles on click", async () => {
    serve([...groupChats(2), ...groupChats(2, mariaExpert.id)]);
    renderRecentChats();

    expect(await screen.findByText("autopilot chat 1")).toBeDefined();
    expect(screen.getByText("expert-maria chat 1")).toBeDefined();

    // The header's name comes from the expert list, a separate request.
    fireEvent.click(await screen.findByRole("button", { name: "Maria chats" }));
    await waitFor(() =>
      expect(screen.queryByText("expert-maria chat 1")).toBeNull(),
    );
    fireEvent.click(groupHeader("Maria"));
    expect(await screen.findByText("expert-maria chat 1")).toBeDefined();

    fireEvent.click(groupHeader("Otto"));
    await waitFor(() =>
      expect(screen.queryByText("autopilot chat 1")).toBeNull(),
    );
  });

  it("keeps a running chat visible while its group is collapsed", async () => {
    serve([
      { id: "running", title: "running chat", is_processing: true },
      ...groupChats(2),
    ]);
    renderRecentChats();
    fireEvent.click(await screen.findByRole("button", { name: "Otto chats" }));

    await waitFor(() =>
      expect(screen.queryByText("autopilot chat 1")).toBeNull(),
    );
    expect(screen.getByText("running chat")).toBeDefined();
  });

  it("fills the sidebar to 30 chats and pages ten more per Show more", async () => {
    serve(groupChats(45));
    renderRecentChats();

    expect(await screen.findByText("autopilot chat 30")).toBeDefined();
    expect(screen.queryByText("autopilot chat 31")).toBeNull();

    await showMore("Otto");
    expect(await screen.findByText("autopilot chat 40")).toBeDefined();
    expect(screen.queryByText("autopilot chat 41")).toBeNull();
    const page = requests.find((request) => request.path === "sessions");
    expect(page?.params.get("autopilot_only")).toBe("true");
    expect(page?.params.get("pinned_first")).toBe("false");
    expect(page?.params.has("expert_id")).toBe(false);

    await showMore("Otto");
    expect(await screen.findByText("autopilot chat 45")).toBeDefined();
    expect(
      screen.queryByRole("button", { name: "Show more Otto chats" }),
    ).toBeNull();
  });

  it("gives the experts in use more of the sidebar, but every expert three", async () => {
    // Otto's 40 chats are the newest, Maria's 10 and Clip's 5 older: each
    // keeps three, and the 21 slots left go to the newest of the rest.
    serve([
      ...groupChats(40),
      ...groupChats(10, mariaExpert.id),
      ...groupChats(5, "expert-clip"),
    ]);
    renderRecentChats();

    expect(await screen.findByText("autopilot chat 24")).toBeDefined();
    expect(screen.queryByText("autopilot chat 25")).toBeNull();
    expect(screen.getByText("expert-maria chat 3")).toBeDefined();
    expect(screen.queryByText("expert-maria chat 4")).toBeNull();
    expect(screen.getByText("expert-clip chat 3")).toBeDefined();
    expect(screen.queryByText("expert-clip chat 4")).toBeNull();
  });

  it("counts every chat in a group, not just the ones listed", async () => {
    serve([...groupChats(3), ...groupChats(40, mariaExpert.id)]);
    renderRecentChats();

    const mariaHeader = await screen.findByRole("button", {
      name: "Maria chats",
    });
    expect(within(mariaHeader).getByText("40")).toBeDefined();
    const description = mariaHeader.getAttribute("aria-describedby");
    expect(
      description && document.getElementById(description)?.textContent,
    ).toBe("40");
    expect(within(groupHeader("Otto")).getByText("3")).toBeDefined();
    expect(screen.getByText("expert-maria chat 27")).toBeDefined();
    expect(screen.queryByText("expert-maria chat 28")).toBeNull();
  });

  it("lists an expert whose chats are all older than everyone else's", async () => {
    // Fifty newer Otto chats used to push Maria off the first page, and with
    // her the whole group, until the list-wide Load more was pressed.
    serve([...groupChats(50), ...groupChats(2, mariaExpert.id)]);
    renderRecentChats();

    expect(await screen.findByText("expert-maria chat 1")).toBeDefined();
    expect(within(groupHeader("Otto")).getByText("50")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Load more" })).toBeNull();
  });

  it("pages an expert's older chats from that expert alone", async () => {
    // Otto's 12 are the newest, so Maria shows 18 of her 40 at first.
    serve([...groupChats(12), ...groupChats(40, mariaExpert.id)]);
    renderRecentChats();

    expect(await screen.findByText("Marketing Manager")).toBeDefined();
    expect(screen.getByText("expert-maria chat 18")).toBeDefined();
    expect(screen.queryByText("expert-maria chat 19")).toBeNull();
    expect(
      screen.getByText("Marketing Manager").classList.contains("opacity-70"),
    ).toBe(true);
    expect(screen.queryByText(mariaExpert.role)).toBeNull();

    await showMore("Maria");
    expect(await screen.findByText("expert-maria chat 28")).toBeDefined();
    expect(screen.queryByText("expert-maria chat 29")).toBeNull();
    expect(screen.queryByText("autopilot chat 13")).toBeNull();
    const page = requests.find((request) => request.path === "sessions");
    expect(page?.params.get("expert_id")).toBe(mariaExpert.id);
    expect(page?.params.has("autopilot_only")).toBe(false);
  });

  it("remembers a collapsed group the next time the sidebar mounts", async () => {
    serve([...groupChats(1), ...groupChats(1, mariaExpert.id)]);
    const { unmount } = renderRecentChats();
    fireEvent.click(await screen.findByRole("button", { name: "Maria chats" }));
    await waitFor(() =>
      expect(screen.queryByText("expert-maria chat 1")).toBeNull(),
    );
    unmount();

    renderRecentChats();
    expect(await screen.findByText("autopilot chat 1")).toBeDefined();
    expect(
      await screen.findByRole("button", { name: "Maria chats" }),
    ).toBeDefined();
    expect(screen.queryByText("expert-maria chat 1")).toBeNull();
  });

  it("falls back to a generic Expert label when the expert is unknown", async () => {
    serve(groupChats(2, "expert-ghost"), []);
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
    serve(groupChats(2, novaExpert.id), [novaExpert]);
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

  it("keeps a group's shown chats and collapse across the 10-second refresh", async () => {
    serve([...groupChats(45), ...groupChats(2, mariaExpert.id)]);
    renderRecentChats();

    fireEvent.click(await screen.findByRole("button", { name: "Maria chats" }));
    await showMore("Otto");
    expect(await screen.findByText("autopilot chat 38")).toBeDefined();

    const reads = () =>
      requests.filter((request) => request.path === "session-groups").length;
    const readsBefore = reads();
    await act(async () => {
      vi.advanceTimersByTime(SESSION_LIST_REFETCH_INTERVAL_MS);
    });
    await waitFor(() => expect(reads()).toBeGreaterThan(readsBefore));

    expect(screen.getByText("autopilot chat 38")).toBeDefined();
    expect(screen.queryByText("expert-maria chat 1")).toBeNull();
  });

  it("links each group header to a new chat, except for fired experts", async () => {
    const maxExpert: Expert = {
      ...mariaExpert,
      id: "expert-max",
      name: "Max",
      is_archived: true,
    };
    serve(
      [
        ...groupChats(1),
        ...groupChats(1, mariaExpert.id),
        ...groupChats(1, maxExpert.id),
      ],
      [mariaExpert, maxExpert],
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
    serve([
      {
        id: "maria-running",
        title: "maria running chat",
        expert_id: mariaExpert.id,
        is_processing: true,
      },
      ...groupChats(1, mariaExpert.id),
    ]);
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
    serve([...groupChats(1), ...groupChats(1, mariaExpert.id)]);
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
