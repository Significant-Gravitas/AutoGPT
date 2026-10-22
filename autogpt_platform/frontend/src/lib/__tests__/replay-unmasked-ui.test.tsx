import { getGetV2ListSessionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { getListExpertIdentitiesMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { LibraryAgentCard } from "@/app/(platform)/library/components/LibraryAgentCard/LibraryAgentCard";
import { LibraryEmptyState } from "@/app/(platform)/library/components/LibraryEmptyState/LibraryEmptyState";
import { FavoriteAnimationProvider } from "@/app/(platform)/library/context/FavoriteAnimationContext";
import type { AgentStatusInfo } from "@/app/(platform)/library/types";
import { SettingsSidebar } from "@/app/(platform)/settings/components/SettingsSidebar/SettingsSidebar";
import { Message, MessageContent } from "@/components/ai-elements/message";
import { AppSidebar } from "@/components/layout/AppSidebar/AppSidebar";
import { AccountMenu } from "@/components/layout/Navbar/components/AccountMenu/AccountMenu";
import { AccountMenuHeader } from "@/components/layout/Navbar/components/AccountMenu/components/AccountMenuHeader";
import { getAccountMenuItems } from "@/components/layout/Navbar/helpers";
import { SidebarProvider } from "@/components/ui/sidebar";
import { ComposerPlusMenu } from "@/app/(platform)/copilot/components/ChatInput/components/ComposerPlusMenu";
import { server } from "@/mocks/mock-server";
import {
  UnmaskedContainer,
  recordReplayText,
} from "@/tests/integrations/sentry-replay";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { cleanup } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const USER_NAME = "Alice Example";
const USER_EMAIL = "alice@example.com";
const CHAT_TITLE = "Plan my trip to Lisbon";
const WORKFLOW_NAME = "Quarterly invoice digest";

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "user-1", email: USER_EMAIL, role: "user" },
    isLoggedIn: true,
    isUserLoading: false,
  }),
}));

vi.mock("next/link", () => ({
  __esModule: true,
  default: ({
    children,
    href,
    ...props
  }: {
    children: ReactNode;
    href: string;
  }) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
  useLinkStatus: () => ({ pending: false }),
}));

vi.mock("next/navigation", async (importOriginal) => {
  const actual = await importOriginal<typeof import("next/navigation")>();
  return {
    ...actual,
    useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
    usePathname: () => "/settings/profile",
    useSearchParams: () => new URLSearchParams(),
  };
});

// Renders popover content in place, so the account menu is on screen.
vi.mock("@/components/molecules/Popover/Popover", () => {
  function Passthrough({ children }: { children: ReactNode }) {
    return <div>{children}</div>;
  }
  return {
    Popover: Passthrough,
    PopoverTrigger: Passthrough,
    PopoverContent: Passthrough,
  };
});

function inUnmaskedContainer(children: ReactNode) {
  return <UnmaskedContainer>{children}</UnmaskedContainer>;
}

beforeEach(() => {
  server.use(
    getListExpertIdentitiesMockHandler([]),
    getGetV2ListSessionsMockHandler200({
      sessions: [
        {
          id: "chat-1",
          title: CHAT_TITLE,
          is_processing: false,
          is_pinned: false,
          created_at: "2026-10-22T10:00:00",
          updated_at: "2026-10-22T10:00:00",
        },
      ],
      total: 1,
    }),
  );
});

afterEach(() => {
  cleanup();
  server.resetHandlers();
});

describe("what Sentry replays show of the app", () => {
  it("shows the sidebar navigation but not the user's chats", async () => {
    render(
      <SidebarProvider>
        <AppSidebar />
      </SidebarProvider>,
    );
    await screen.findByText(CHAT_TITLE);

    const recorded = await recordReplayText();
    for (const label of [
      "Home",
      "Marketplace",
      "Build",
      "Workspace",
      "Team",
      "Files",
      "Recent chats",
    ]) {
      expect(recorded).toContain(label);
    }
    expect(recorded).not.toContain(CHAT_TITLE);
    expect(recorded).not.toContain(USER_EMAIL);
  });

  it("shows a chat row's actions but not the chat's title", async () => {
    render(
      <SidebarProvider>
        <AppSidebar />
      </SidebarProvider>,
    );
    await screen.findByText(CHAT_TITLE);
    fireEvent.pointerDown(
      screen.getByRole("button", { name: "Chat actions" }),
      {
        button: 0,
      },
    );
    await screen.findByRole("menuitem", { name: "Rename" });

    const recorded = await recordReplayText();
    for (const label of [
      "Rename",
      "Export chat",
      "Share chat",
      "Delete chat",
    ]) {
      expect(recorded).toContain(label);
    }
    expect(recorded).not.toContain(CHAT_TITLE);
  });

  it("shows the composer's + button and menu", async () => {
    render(<ComposerPlusMenu onFilesSelected={vi.fn()} />);
    expect(await recordReplayText()).toContain("Add files and more");

    fireEvent.pointerDown(screen.getByTestId("composer-plus-button"), {
      button: 0,
    });
    await screen.findByRole("menuitem", { name: "Attach file" });

    const recorded = await recordReplayText();
    for (const label of [
      "Attach file",
      "Connect service",
      "Skills",
      "Scheduled",
    ]) {
      expect(recorded).toContain(label);
    }
  });

  it("shows account menu items but not who is signed in", async () => {
    render(
      <AccountMenu
        userName={USER_NAME}
        userHandle="alice"
        userEmail={USER_EMAIL}
        menuItemGroups={getAccountMenuItems("user", true)}
      />,
    );
    await screen.findByText(USER_EMAIL);

    const recorded = await recordReplayText();
    for (const label of ["Profile", "Settings", "Billing", "Log out"]) {
      expect(recorded).toContain(label);
    }
    expect(recorded).not.toContain(USER_NAME);
    expect(recorded).not.toContain(USER_EMAIL);
    expect(recorded).not.toContain("alice");
  });

  it("shows the settings navigation", async () => {
    render(<SettingsSidebar />);

    const recorded = await recordReplayText();
    for (const label of ["Back", "Profile", "Account", "Integrations"]) {
      expect(recorded).toContain(label);
    }
  });

  it("shows the empty library's fixed copy", async () => {
    render(<LibraryEmptyState />);

    const recorded = await recordReplayText();
    expect(recorded).toContain("Your library is empty");
    expect(recorded).toContain("Build a workflow");
    expect(recorded).toContain("Browse marketplace");
  });

  describe("keeps user content masked inside an unmasked container", () => {
    it("chat messages, from the user and from Otto", async () => {
      render(
        inUnmaskedContainer(
          <>
            <Message from="user">
              <MessageContent>{CHAT_TITLE}</MessageContent>
            </Message>
            <Message from="assistant">
              <MessageContent>
                <p>
                  Booked your flight to <strong>Lisbon</strong>.
                </p>
              </MessageContent>
            </Message>
          </>,
        ),
      );

      const recorded = await recordReplayText();
      expect(recorded).not.toContain("Plan my trip");
      expect(recorded).not.toContain("Booked your flight");
      expect(recorded).not.toContain("Lisbon");
    });

    it("the signed-in user's name and email", async () => {
      render(
        inUnmaskedContainer(
          <AccountMenuHeader userName={USER_NAME} userEmail={USER_EMAIL} />,
        ),
      );
      await screen.findByText(USER_EMAIL);

      const recorded = await recordReplayText();
      expect(recorded).not.toContain(USER_NAME);
      expect(recorded).not.toContain(USER_EMAIL);
    });

    it("a workflow's name", async () => {
      const agent = {
        id: "4f6b2c1e-0000-4000-8000-000000000001",
        name: WORKFLOW_NAME,
        image_url: "https://cdn.test/agent.png",
        is_favorite: false,
      } as unknown as LibraryAgent;
      const status: AgentStatusInfo = {
        status: "idle",
        health: "good",
        progress: null,
        totalRuns: 3,
        lastRunAt: null,
        lastError: null,
        activeExecutionID: null,
        monthlySpend: 0,
        nextScheduledRun: null,
        triggerType: null,
      };
      render(
        inUnmaskedContainer(
          <FavoriteAnimationProvider>
            <LibraryAgentCard agent={agent} statusInfo={status} />
          </FavoriteAnimationProvider>,
        ),
      );
      await screen.findByText(WORKFLOW_NAME);

      expect(await recordReplayText()).not.toContain(WORKFLOW_NAME);
    });

    it("a chat title in the sidebar", async () => {
      render(
        inUnmaskedContainer(
          <SidebarProvider>
            <AppSidebar />
          </SidebarProvider>,
        ),
      );
      await screen.findByText(CHAT_TITLE);

      expect(await recordReplayText()).not.toContain(CHAT_TITLE);
    });
  });
});
