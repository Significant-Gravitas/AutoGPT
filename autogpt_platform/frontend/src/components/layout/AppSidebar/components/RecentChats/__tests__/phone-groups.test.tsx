import { getListExpertIdentitiesMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { SidebarProvider } from "@/components/molecules/Sidebar/Sidebar";
import { server } from "@/mocks/mock-server";
import { Key, storage } from "@/services/storage/local-storage";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { RecentChats } from "../RecentChats";
import {
  type SidebarChat,
  type SidebarRequest,
  groupChats,
  sidebarBackend,
} from "./sidebarBackend";

// In the phone menu the whole budget would bury every group below Otto's, so
// each group opens on its newest four and the other experts stay in view.
vi.mock("@/hooks/use-mobile", () => ({ useIsMobile: () => true }));

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

async function showMoreOtto() {
  fireEvent.click(
    await screen.findByRole("button", { name: "Show more Otto chats" }),
  );
}

beforeEach(() => {
  storage.clean(Key.RECENT_CHATS_COLLAPSED_GROUPS);
  requests = [];
  // Otto's 30 chats are the newest, so the grouped read gives Otto 27 of its
  // 30 slots and Maria her minimum of 3.
  chats = [...groupChats(30), ...groupChats(6, mariaExpert.id)];
  server.use(
    ...sidebarBackend(() => chats, requests),
    getListExpertIdentitiesMockHandler([mariaExpert]),
  );
});

afterEach(() => {
  server.resetHandlers();
});

describe("RecentChats — phone menu", () => {
  it("opens Otto on its newest four, so the next expert's chats show too", async () => {
    renderRecentChats();

    expect(await screen.findByText("autopilot chat 4")).toBeDefined();
    expect(screen.queryByText("autopilot chat 5")).toBeNull();
    expect(screen.getByText("expert-maria chat 3")).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Show more Otto chats" }),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Show more Maria chats" }),
    ).toBeDefined();
  });

  it("shows the chats the grouped read already has before reading older ones", async () => {
    renderRecentChats();
    await screen.findByText("autopilot chat 4");

    await showMoreOtto();
    expect(await screen.findByText("autopilot chat 14")).toBeDefined();
    expect(screen.queryByText("autopilot chat 15")).toBeNull();
    expect(reads("sessions")).toBe(0);

    // Past the 27 the grouped read listed, the group reads its older chats.
    await showMoreOtto();
    await showMoreOtto();
    expect(await screen.findByText("autopilot chat 30")).toBeDefined();
    expect(reads("sessions")).toBeGreaterThan(0);
  });
});
