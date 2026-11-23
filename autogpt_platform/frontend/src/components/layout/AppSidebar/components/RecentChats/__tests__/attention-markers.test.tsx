import { getListExpertIdentitiesMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { SidebarProvider } from "@/components/molecules/Sidebar/Sidebar";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, it } from "vitest";

import { RecentChats } from "../RecentChats";
import {
  type SidebarChat,
  type SidebarRequest,
  sidebarBackend,
} from "./sidebarBackend";

// SECRT-3542: which chats want the user comes from the server, so every
// browser and device shows the same markers: needs you, error and unread.

const maria = {
  id: "expert-maria",
  name: "Maria",
  avatar_url: null,
  role: "Marketing Strategist",
  job_title: "Marketing Manager",
  bio: null,
  skills: [],
  tagline: null,
  identity: "You are Maria.",
  voice_preferences: null,
  boundaries: null,
  protected_soul_rules: [],
  is_template: false,
  source_template_id: null,
  is_archived: false,
  workflows: [],
} as unknown as Expert;

let chats: SidebarChat[] = [];
let requests: SidebarRequest[] = [];

beforeEach(() => {
  requests = [];
  chats = [
    { id: "otto-1", title: "Weekly sales summary", attention: "unread" },
    { id: "otto-2", title: "Plan the offsite", is_processing: true },
    { id: "otto-3", title: "Draft investor update" },
    {
      id: "maria-1",
      title: "Q4 campaign ideas",
      expert_id: maria.id,
      attention: "needs_you",
    },
    {
      id: "maria-2",
      title: "Newsletter copy",
      expert_id: maria.id,
      attention: "unread",
    },
    {
      id: "maria-3",
      title: "Post the launch thread",
      expert_id: maria.id,
      attention: "error",
    },
  ];
  server.use(
    ...sidebarBackend(() => chats, requests),
    getListExpertIdentitiesMockHandler([maria]),
  );
});

function renderRecentChats() {
  return render(
    <SidebarProvider>
      <RecentChats />
    </SidebarProvider>,
  );
}

function readBodies() {
  return requests
    .filter((request) => request.path === "sessions/read")
    .map((request) => request.body);
}

function row(title: string) {
  return screen.getByRole("link", { name: new RegExp(title) });
}

describe("RecentChats — what each chat wants from the user", () => {
  it("marks each chat as the server reports it, one marker at a time", async () => {
    renderRecentChats();
    await screen.findByText("Q4 campaign ideas");

    expect(
      within(row("Q4 campaign ideas")).getByRole("img", { name: "Needs you" }),
    ).toBeDefined();
    expect(
      within(row("Post the launch thread")).getByRole("img", {
        name: "Stopped with an error",
      }),
    ).toBeDefined();
    expect(
      within(row("Newsletter copy")).getByRole("img", { name: "Unread" }),
    ).toBeDefined();
    expect(
      within(row("Weekly sales summary")).getByRole("img", { name: "Unread" }),
    ).toBeDefined();
    expect(within(row("Draft investor update")).queryByRole("img")).toBeNull();
    // A running chat shows its spinner, not a marker.
    expect(within(row("Plan the offsite")).queryByRole("img")).toBeNull();
  });

  it("counts each expert's chats that want the user, for screen readers too", async () => {
    renderRecentChats();
    await screen.findByText("Q4 campaign ideas");

    const maria = screen.getByRole("button", { name: "Maria chats" });
    expect(
      within(maria).getByText("Needs attention:").parentElement?.textContent,
    ).toBe("Needs attention:3");
    const description = (maria.getAttribute("aria-describedby") ?? "")
      .split(" ")
      .map((id) => document.getElementById(id)?.textContent)
      .join(" ");
    expect(description).toBe("3 Needs attention:3");
  });
});

describe("RecentChats — marking an expert's chats read", () => {
  it("clears the unread and failed chats on the server and keeps what needs the user", async () => {
    renderRecentChats();
    await screen.findByText("Q4 campaign ideas");

    fireEvent.click(
      screen.getByRole("button", { name: "Mark all Maria chats as read" }),
    );

    await waitFor(() =>
      expect(
        within(row("Newsletter copy")).queryByRole("img", { name: "Unread" }),
      ).toBeNull(),
    );
    expect(readBodies()).toEqual([{ expert_id: maria.id }]);
    expect(within(row("Post the launch thread")).queryByRole("img")).toBeNull();
    expect(
      within(row("Q4 campaign ideas")).getByRole("img", { name: "Needs you" }),
    ).toBeDefined();
    // Otto's chats are another group's.
    expect(
      within(row("Weekly sales summary")).getByRole("img", { name: "Unread" }),
    ).toBeDefined();
    // Nothing left to clear, so the button goes.
    expect(
      screen.queryByRole("button", { name: "Mark all Maria chats as read" }),
    ).toBeNull();
  });

  it("asks for Otto's own chats when marking Otto's group read", async () => {
    renderRecentChats();
    await screen.findByText("Weekly sales summary");

    fireEvent.click(
      screen.getByRole("button", { name: "Mark all Otto chats as read" }),
    );

    await waitFor(() =>
      expect(
        within(row("Weekly sales summary")).queryByRole("img", {
          name: "Unread",
        }),
      ).toBeNull(),
    );
    expect(readBodies()).toEqual([{ autopilot_only: true }]);
    // Maria's chats are another group's.
    expect(
      within(row("Newsletter copy")).getByRole("img", { name: "Unread" }),
    ).toBeDefined();
  });

  it("offers it for a chat that needs the user and also has an unread turn, and keeps its marker", async () => {
    chats = [
      { id: "otto-1", title: "Draft investor update" },
      {
        id: "maria-1",
        title: "Q4 campaign ideas",
        expert_id: maria.id,
        attention: "needs_you",
        unread: true,
      },
    ];
    renderRecentChats();
    await screen.findByText("Q4 campaign ideas");

    fireEvent.click(
      screen.getByRole("button", { name: "Mark all Maria chats as read" }),
    );

    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: "Mark all Maria chats as read" }),
      ).toBeNull(),
    );
    expect(readBodies()).toEqual([{ expert_id: maria.id }]);
    expect(
      within(row("Q4 campaign ideas")).getByRole("img", { name: "Needs you" }),
    ).toBeDefined();
  });

  it("keeps keyboard focus in the group once the button goes", async () => {
    renderRecentChats();
    await screen.findByText("Q4 campaign ideas");

    const button = screen.getByRole("button", {
      name: "Mark all Maria chats as read",
    });
    button.focus();
    fireEvent.click(button);

    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: "Mark all Maria chats as read" }),
      ).toBeNull(),
    );
    expect(document.activeElement).toBe(
      screen.getByRole("button", { name: "Maria chats" }),
    );
  });
});
