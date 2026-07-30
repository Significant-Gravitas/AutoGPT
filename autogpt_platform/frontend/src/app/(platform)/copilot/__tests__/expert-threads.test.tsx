import { getGetV2ListSessionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { getListExpertsMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { SidebarProvider } from "@/components/ui/sidebar";
import { server } from "@/mocks/mock-server";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { parseAsString, useQueryState } from "nuqs";
import { withNuqsTestingAdapter } from "nuqs/adapters/testing";
import { afterEach, describe, expect, it, vi } from "vitest";
import { RecipientChip } from "../components/ChatInput/components/RecipientChip";
import { ChatMessagesContainer } from "../components/ChatMessagesContainer/ChatMessagesContainer";
import { ChatSidebar } from "../components/ChatSidebar/ChatSidebar";
import { useChatSession } from "../useChatSession";
import { useCopilotUIStore } from "../store";
import { groupSessionsByExpert } from "../useSessionList";

const flagState = vi.hoisted(() => ({
  values: { "hire-experts": true } as Record<string, boolean>,
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) => flagState.values[flag] ?? false,
  };
});

vi.mock(
  "../../copilot/components/UsageLimits/UsagePopover/UsagePopover",
  () => ({
    UsagePopover: () => null,
  }),
);
vi.mock(
  "../components/ChatSidebar/components/NotificationToggle/NotificationToggle",
  () => ({
    NotificationToggle: () => null,
  }),
);

vi.mock("use-stick-to-bottom", () => ({
  useStickToBottomContext: () => ({
    scrollRef: { current: { scrollHeight: 100, scrollTop: 0 } },
  }),
  Conversation: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  ConversationContent: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  ConversationScrollButton: () => null,
}));

vi.mock("@/components/ai-elements/conversation", () => ({
  Conversation: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="conversation">{children}</div>
  ),
  ConversationContent: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="conversation-content">{children}</div>
  ),
  ConversationScrollButton: () => null,
}));

vi.mock("@/components/ai-elements/message", () => ({
  Message: ({
    children,
    from,
  }: {
    children: React.ReactNode;
    from?: string;
  }) => <div data-testid={`message-${from ?? "unknown"}`}>{children}</div>,
  MessageActions: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  MessageContent: ({
    children,
    className,
  }: {
    children: React.ReactNode;
    className?: string;
  }) => <div className={className}>{children}</div>,
}));

vi.mock(
  "../components/ChatMessagesContainer/components/AssistantMessageActions",
  () => ({
    AssistantMessageActions: () => null,
  }),
);
vi.mock("../components/ChatMessagesContainer/components/CopyButton", () => ({
  CopyButton: () => null,
}));
vi.mock(
  "../components/ChatMessagesContainer/components/CollapsedToolGroup",
  () => ({
    CollapsedToolGroup: () => null,
  }),
);
vi.mock(
  "../components/ChatMessagesContainer/components/MessageAttachments",
  () => ({
    MessageAttachments: () => null,
  }),
);
vi.mock(
  "../components/ChatMessagesContainer/components/MessagePartRenderer",
  () => ({
    MessagePartRenderer: () => null,
  }),
);
vi.mock("../components/ChatMessagesContainer/components/QueueBadge", () => ({
  QueueBadge: () => null,
}));
vi.mock(
  "../components/ChatMessagesContainer/components/ReasoningGroup",
  () => ({
    ReasoningGroup: () => null,
  }),
);
vi.mock(
  "../components/ChatMessagesContainer/components/ThinkingIndicator",
  () => ({
    ThinkingIndicator: () => null,
  }),
);
vi.mock("../components/ChatMessagesContainer/helpers", () => ({
  buildRenderSegments: () => [],
  getTurnMessages: () => [],
  parseSpecialMarkers: () => ({ markerType: null }),
  shouldShowTaskListNotice: () => false,
  splitReasoningAndResponse: (parts: unknown[]) => ({
    reasoning: [],
    response: parts,
  }),
}));
vi.mock("../components/JobStatsBar/TurnStatsBar", () => ({
  TurnStatsBar: () => null,
}));
vi.mock("../components/JobStatsBar/useElapsedTimer", () => ({
  useElapsedTimer: () => ({ elapsedSeconds: 0 }),
}));
vi.mock("../components/CopilotPendingReviews/CopilotPendingReviews", () => ({
  CopilotPendingReviews: () => null,
}));

const mariaExpert: Expert = {
  id: "expert-maria",
  name: "Maria",
  avatar_url: "https://example.com/maria.png",
  role: "Marketing Strategist",
  bio: null,
  skills: [],
  tagline: "Grows your brand while you sleep",
  identity: "You are Maria, a senior marketing strategist.",
  is_template: false,
  source_template_id: "template-maria",
  is_archived: false,
  workflows: [],
};

function makeSession(overrides: Record<string, unknown>) {
  return {
    id: "s1",
    title: "Chat",
    is_processing: false,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

afterEach(() => {
  cleanup();
  server.resetHandlers();
  flagState.values = { "hire-experts": true };
  useCopilotUIStore.setState({ adoptedExpertThreads: new Set<string>() });
});

function ExpertSessionHarness() {
  const [expertId] = useQueryState("expertId", parseAsString);
  const { createSession, sessionId } = useChatSession({ expertId });
  return (
    <div>
      <div data-testid="session-id">{sessionId ?? "none"}</div>
      <div data-testid="expert-id">{expertId ?? "none"}</div>
      <button onClick={() => void createSession().catch(() => {})}>
        create
      </button>
    </div>
  );
}

/** Mirrors `CopilotPage`, which keys the chat host on the session id — every
 *  session change (including "New Chat" clearing it) remounts `useChatSession`
 *  with fresh refs. */
function KeyedSessionHost() {
  const [sessionId] = useQueryState("sessionId", parseAsString);
  return <ExpertSessionHarness key={`chat-host-${sessionId ?? "new"}`} />;
}

const NuqsWrapper = withNuqsTestingAdapter({
  searchParams: "?expertId=expert-maria",
  hasMemory: true,
});

describe("useChatSession — expert sessions", () => {
  it("creates a session with expert_id when visiting /copilot?expertId=expert-maria", async () => {
    let createBody: unknown = null;
    server.use(
      http.post("*/api/chat/sessions", async ({ request }) => {
        createBody = await request.json();
        return HttpResponse.json({
          id: "new-session-1",
          created_at: "2026-01-01T00:00:00Z",
          user_id: "user-1",
          expert_id: "expert-maria",
        });
      }),
      http.get("*/api/chat/sessions/new-session-1", () =>
        HttpResponse.json({
          id: "new-session-1",
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z",
          user_id: "user-1",
          messages: [],
        }),
      ),
      getGetV2ListSessionsMockHandler200({ sessions: [], total: 0 }),
    );

    render(
      <NuqsWrapper>
        <ExpertSessionHarness />
      </NuqsWrapper>,
    );
    fireEvent.click(screen.getByRole("button", { name: "create" }));

    await waitFor(() => {
      expect(createBody).toEqual({ expert_id: "expert-maria" });
    });
  });

  it("opens the expert's latest thread when one already exists", async () => {
    const seenExpertFilters: (string | null)[] = [];
    server.use(
      http.get("*/api/chat/sessions", ({ request }) => {
        const url = new URL(request.url);
        seenExpertFilters.push(url.searchParams.get("expert_id"));
        return HttpResponse.json({
          sessions: [
            makeSession({
              id: "s-maria-latest",
              title: "Maria thread",
              expert_id: "expert-maria",
              updated_at: "2026-01-02T00:00:00Z",
            }),
          ],
          total: 1,
        });
      }),
      http.get("*/api/chat/sessions/s-maria-latest", () =>
        HttpResponse.json({
          id: "s-maria-latest",
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-02T00:00:00Z",
          user_id: "user-1",
          messages: [],
        }),
      ),
    );

    render(
      <NuqsWrapper>
        <ExpertSessionHarness />
      </NuqsWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByTestId("session-id").textContent).toBe(
        "s-maria-latest",
      );
    });
    expect(seenExpertFilters).toContain("expert-maria");
  });

  it("stays on a fresh session after New Chat instead of re-adopting the expert's thread", async () => {
    server.use(
      http.get("*/api/chat/sessions", ({ request }) => {
        const url = new URL(request.url);
        const expertId = url.searchParams.get("expert_id");
        const mariaSession = makeSession({
          id: "s-maria-latest",
          title: "Maria thread",
          expert_id: "expert-maria",
        });
        if (expertId === "expert-maria") {
          return HttpResponse.json({ sessions: [mariaSession], total: 1 });
        }
        return HttpResponse.json({ sessions: [mariaSession], total: 1 });
      }),
      http.get("*/api/chat/sessions/s-maria-latest", () =>
        HttpResponse.json({
          id: "s-maria-latest",
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-02T00:00:00Z",
          user_id: "user-1",
          expert_id: "expert-maria",
          messages: [],
        }),
      ),
      getListExpertsMockHandler([mariaExpert]),
    );

    render(
      <NuqsWrapper>
        <SidebarProvider>
          <ChatSidebar />
          <KeyedSessionHost />
        </SidebarProvider>
      </NuqsWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByTestId("session-id").textContent).toBe(
        "s-maria-latest",
      );
    });

    // Click inside waitFor, re-querying each attempt: the sidebar re-renders
    // as the adoption lands and the session list arrives, which can replace
    // the button node between a one-shot query and its click (a click on the
    // detached node is a no-op). Re-clicking is idempotent (sets null).
    await waitFor(() => {
      fireEvent.click(screen.getAllByRole("button", { name: "New Chat" })[0]);
      expect(screen.getByTestId("session-id").textContent).toBe("none");
    });
    // The remount re-runs the adoption effect against a warm cache, so give it
    // a chance to bounce back before declaring the fix good.
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(screen.getByTestId("session-id").textContent).toBe("none");
    expect(screen.getByTestId("expert-id").textContent).toBe("none");
  });
});

describe("groupSessionsByExpert", () => {
  it("partitions sessions by expert_id with the Autopilot group first", () => {
    const groups = groupSessionsByExpert([
      makeSession({ id: "s1" }),
      makeSession({ id: "s2", expert_id: "expert-maria" }),
      makeSession({ id: "s3", expert_id: "expert-juan" }),
      makeSession({ id: "s4", expert_id: "expert-maria" }),
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ] as any);

    expect(groups.map((group) => group.expertId)).toEqual([
      null,
      "expert-maria",
      "expert-juan",
    ]);
    expect(
      groups[1].sessions.map((session: { id: string }) => session.id),
    ).toEqual(["s2", "s4"]);
    expect(
      groups[0].sessions.map((session: { id: string }) => session.id),
    ).toEqual(["s1"]);
  });
});

describe("ChatSidebar — expert groups", () => {
  it("groups expert threads under expert name headers with Autopilot as the default group", async () => {
    server.use(
      getGetV2ListSessionsMockHandler200({
        sessions: [
          makeSession({ id: "s1", title: "Plain chat" }),
          makeSession({
            id: "s2",
            title: "Campaign ideas",
            expert_id: "expert-maria",
          }),
        ],
        total: 2,
      }),
      getListExpertsMockHandler([mariaExpert]),
    );

    render(
      <SidebarProvider>
        <ChatSidebar />
      </SidebarProvider>,
    );

    await screen.findByText("Plain chat");
    const mariaHeader = await screen.findByTestId(
      "expert-group-header-expert-maria",
    );
    expect(mariaHeader.textContent).toBe("Maria");
    expect(
      screen.getByTestId("expert-group-header-autopilot").textContent,
    ).toBe("Autopilot");
    expect(screen.getByText("Campaign ideas")).toBeDefined();
  });

  it("renders no group headers when the user has no expert threads", async () => {
    server.use(
      getGetV2ListSessionsMockHandler200({
        sessions: [
          makeSession({ id: "s1", title: "Plain chat" }),
          makeSession({ id: "s2", title: "Another plain chat" }),
        ],
        total: 2,
      }),
      getListExpertsMockHandler([]),
    );

    render(
      <SidebarProvider>
        <ChatSidebar />
      </SidebarProvider>,
    );

    await screen.findByText("Plain chat");
    expect(screen.getByText("Another plain chat")).toBeDefined();
    expect(screen.queryByTestId("expert-group-header-autopilot")).toBeNull();
  });

  it("keeps pinned chats above the groups so pinning still floats expert threads", async () => {
    flagState.values = { "hire-experts": true, "chat-pinning": true };
    server.use(
      getGetV2ListSessionsMockHandler200({
        sessions: [
          makeSession({
            id: "s-pinned",
            title: "Pinned campaign",
            expert_id: "expert-maria",
            is_pinned: true,
          }),
          makeSession({ id: "s-plain", title: "Plain chat" }),
          makeSession({
            id: "s-maria",
            title: "Campaign ideas",
            expert_id: "expert-maria",
          }),
        ],
        total: 3,
      }),
      getListExpertsMockHandler([mariaExpert]),
    );

    render(
      <SidebarProvider>
        <ChatSidebar />
      </SidebarProvider>,
    );

    const pinnedSection = await screen.findByTestId(
      "expert-group-header-pinned",
    );
    expect(pinnedSection.textContent).toBe("Pinned");

    const pinnedGroup = pinnedSection.closest('[role="group"]');
    expect(pinnedGroup).not.toBeNull();
    expect(within(pinnedGroup as HTMLElement).getByText("Pinned campaign"));

    // The pinned chat is lifted out of Maria's group, which keeps the rest.
    const mariaGroup = screen
      .getByTestId("expert-group-header-expert-maria")
      .closest('[role="group"]') as HTMLElement;
    expect(within(mariaGroup).getByText("Campaign ideas")).toBeDefined();
    expect(within(mariaGroup).queryByText("Pinned campaign")).toBeNull();
  });

  it("does not fetch experts or group sessions when the flag is off", async () => {
    flagState.values = { "hire-experts": false };
    let expertsRequests = 0;
    server.use(
      http.get("*/api/experts", () => {
        expertsRequests += 1;
        return HttpResponse.json([mariaExpert]);
      }),
      getGetV2ListSessionsMockHandler200({
        sessions: [
          makeSession({ id: "s1", title: "Plain chat" }),
          makeSession({
            id: "s2",
            title: "Campaign ideas",
            expert_id: "expert-maria",
          }),
        ],
        total: 2,
      }),
    );

    render(
      <SidebarProvider>
        <ChatSidebar />
      </SidebarProvider>,
    );

    await screen.findByText("Plain chat");
    expect(screen.getByText("Campaign ideas")).toBeDefined();
    expect(screen.queryByTestId("expert-group-header-autopilot")).toBeNull();
    expect(screen.queryByTestId("expert-group-header-expert-maria")).toBeNull();
    expect(expertsRequests).toBe(0);
  });
});

describe("ChatMessagesContainer — expert identity", () => {
  const assistantMessage = {
    id: "m1",
    role: "assistant",
    parts: [{ type: "text", text: "Here is your marketing plan." }],
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  } as any;

  it("shows the expert name and avatar in the thread header and on assistant messages", () => {
    render(
      <ChatMessagesContainer
        messages={[assistantMessage]}
        status="ready"
        error={undefined}
        isLoading={false}
        expertIdentity={{
          name: "Maria",
          avatarUrl: mariaExpert.avatar_url,
          role: mariaExpert.role,
        }}
      />,
    );

    const header = screen.getByTestId("expert-thread-header");
    expect(within(header).getByText("Maria")).toBeDefined();
    expect(within(header).getByRole("img", { name: "Maria" })).toBeDefined();
    const assistantIdentity = screen.getByTestId("expert-assistant-identity");
    expect(within(assistantIdentity).getByText("Maria")).toBeDefined();
  });

  it("renders no expert header or identity for plain sessions", () => {
    render(
      <ChatMessagesContainer
        messages={[assistantMessage]}
        status="ready"
        error={undefined}
        isLoading={false}
      />,
    );

    expect(screen.queryByTestId("expert-thread-header")).toBeNull();
    expect(screen.queryByTestId("expert-assistant-identity")).toBeNull();
  });
});

describe("recipient picker", () => {
  it("does not look up or adopt the expert's latest thread when the recipient is picked after mount", async () => {
    let expertListRequests = 0;
    server.use(
      http.get("*/api/chat/sessions", ({ request }) => {
        const url = new URL(request.url);
        if (url.searchParams.get("expert_id") === "expert-maria") {
          expertListRequests += 1;
          return HttpResponse.json({
            sessions: [
              makeSession({ id: "old-thread", expert_id: "expert-maria" }),
            ],
            total: 1,
          });
        }
        return HttpResponse.json({ sessions: [], total: 0 });
      }),
    );

    function RecipientSwitchHarness() {
      const [expertId, setExpertId] = useQueryState("expertId", parseAsString);
      const { sessionId } = useChatSession({ expertId });
      return (
        <div>
          <div data-testid="session-id">{sessionId ?? "none"}</div>
          <div data-testid="expert-id">{expertId ?? "none"}</div>
          <button onClick={() => void setExpertId("expert-maria")}>
            pick maria
          </button>
        </div>
      );
    }

    const FreshMountWrapper = withNuqsTestingAdapter({
      searchParams: "?",
      hasMemory: true,
    });
    render(
      <FreshMountWrapper>
        <RecipientSwitchHarness />
      </FreshMountWrapper>,
    );

    fireEvent.click(screen.getByRole("button", { name: "pick maria" }));
    await waitFor(() =>
      expect(screen.getByTestId("expert-id").textContent).toBe("expert-maria"),
    );
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(screen.getByTestId("session-id").textContent).toBe("none");
    // Adoption is impossible here, so the lookup should never be issued.
    expect(expertListRequests).toBe(0);
  });

  it("RecipientChip shows a placeholder instead of Autopilot while experts load", () => {
    render(
      <RecipientChip
        recipient={{ id: null, name: "Autopilot", avatarUrl: null }}
        options={[]}
        onSelect={vi.fn()}
        isLoading
      />,
    );

    expect(screen.getByRole("status", { name: "Loading recipient" }));
    expect(screen.queryByText("Autopilot")).toBeNull();
  });

  it("RecipientChip lists the team and reports the selection", async () => {
    const onSelect = vi.fn();
    render(
      <RecipientChip
        recipient={{ id: null, name: "Autopilot", avatarUrl: null }}
        options={[
          { id: null, name: "Autopilot", avatarUrl: null },
          { id: "expert-maria", name: "Maria", avatarUrl: null },
        ]}
        onSelect={onSelect}
      />,
    );

    await userEvent.click(
      screen.getByRole("button", { name: /change recipient/i }),
    );
    await userEvent.click(await screen.findByText("Maria"));
    expect(onSelect).toHaveBeenCalledWith("expert-maria");
  });
});
