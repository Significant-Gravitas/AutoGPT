import { getGetV2ListSessionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { getListExpertIdentitiesMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { SidebarProvider } from "@/components/molecules/Sidebar/Sidebar";
import { server } from "@/mocks/mock-server";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { RecentChats } from "../RecentChats";

const location = vi.hoisted(() => ({ pathname: "/home", search: "" }));

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: () => location.pathname,
  useSearchParams: () => new URLSearchParams(location.search),
  useParams: () => ({}),
}));

const maria: Expert = {
  id: "expert-maria",
  name: "Maria",
  avatar_url: null,
  role: "Marketing Strategist",
  job_title: "Marketing Manager",
  bio: null,
  skills: [],
  tagline: null,
  identity: "",
  voice_preferences: "",
  boundaries: "",
  protected_soul_rules: [],
  is_template: false,
  source_template_id: "template-maria",
  is_archived: false,
  workflows: [],
};

function makeSession(id: string, expertId: string | null) {
  return {
    id,
    title: `${id} chat`,
    is_processing: false,
    created_at: "2026-06-30T10:00:00",
    updated_at: "2026-06-30T10:00:00",
    expert_id: expertId,
  };
}

const sessions = [
  makeSession("otto-1", null),
  makeSession("maria-1", maria.id),
  makeSession("maria-2", maria.id),
];

function sidebar() {
  return (
    <SidebarProvider>
      <RecentChats />
    </SidebarProvider>
  );
}

beforeEach(() => {
  location.pathname = "/home";
  location.search = "";
  server.use(
    getGetV2ListSessionsMockHandler200({ sessions, total: sessions.length }),
    getListExpertIdentitiesMockHandler([maria]),
  );
});

describe("RecentChats — the group of the chat on screen", () => {
  it("opens with the chat on screen, and stays shut once the user closes it", async () => {
    location.search = "sessionId=maria-2";
    render(sidebar());

    expect(await screen.findByText("maria-1 chat")).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Maria chats" }));
    expect(screen.queryByText("maria-1 chat")).toBeNull();
  });

  it("reopens a group the user closed when another of its chats comes on screen", async () => {
    location.search = "sessionId=maria-1";
    const { rerender } = render(sidebar());
    expect(await screen.findByText("maria-2 chat")).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Maria chats" }));
    expect(screen.queryByText("maria-2 chat")).toBeNull();

    location.search = "sessionId=maria-2";
    rerender(sidebar());
    expect(await screen.findByText("maria-1 chat")).toBeDefined();
  });

  it("opens when one of the expert's chats is opened later", async () => {
    location.search = "sessionId=otto-1";
    const { rerender } = render(sidebar());
    await screen.findByRole("button", { name: "Maria chats" });
    expect(screen.queryByText("maria-1 chat")).toBeNull();

    location.search = "sessionId=maria-1";
    rerender(sidebar());
    expect(await screen.findByText("maria-2 chat")).toBeDefined();
  });
});
