import React from "react";
import { getGetV2GetChatShareStateMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { ChatShareStateResponse } from "@/app/api/__generated__/models/chatShareStateResponse";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  cleanup,
  fireEvent,
  waitFor,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ChatContainer } from "../ChatContainer";
import { useCopilotUIStore } from "../../../store";

const mockIsUsageLimitReached = vi.fn();
const mockArtifactsEnabled = vi.fn(() => false);
const clipboardWrite = vi.fn(async (_text: string) => {});

const ARTIFACT_A_ID = "11111111-0000-0000-0000-000000000000";
const ARTIFACT_B_ID = "22222222-0000-0000-0000-000000000000";
const SHARED_TOKEN = "33333333-0000-0000-0000-000000000000";

function makeArtifact(id: string, title = `${id}.txt`) {
  return {
    id,
    title,
    mimeType: "text/plain",
    sourceUrl: `/api/proxy/api/workspace/files/${id}/download`,
    origin: "agent" as const,
  };
}

function resetCopilotStore() {
  useCopilotUIStore.setState({
    artifactPanel: {
      isOpen: false,
      activeArtifact: null,
      history: [],
      activeTab: "files",
    },
  });
}

function mockShareState(state: Partial<ChatShareStateResponse>) {
  server.use(
    getGetV2GetChatShareStateMockHandler200(
      (): ChatShareStateResponse => ({
        is_shared: false,
        share_token: null,
        auto_share_executions: false,
        message_count: 0,
        linked_run_count: 0,
        file_count: 0,
        ...state,
      }),
    ),
  );
}

vi.mock("framer-motion", () => ({
  LayoutGroup: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  motion: {
    div: React.forwardRef(function MotionDiv(
      props: Record<string, unknown>,
      ref: React.Ref<HTMLDivElement>,
    ) {
      const {
        children,
        initial: _initial,
        animate: _animate,
        transition: _transition,
        ...rest
      } = props as {
        children?: React.ReactNode;
        initial?: unknown;
        animate?: unknown;
        transition?: unknown;
        [key: string]: unknown;
      };

      return (
        <div ref={ref} {...rest}>
          {children}
        </div>
      );
    }),
  },
}));

vi.mock("@/app/(platform)/copilot/components/ChatInput/ChatInput", () => ({
  ChatInput: () => <div data-testid="chat-input" />,
}));

vi.mock("@/components/atoms/Tooltip/BaseTooltip", () => ({
  TooltipProvider: ({ children }: { children: React.ReactNode }) => (
    <>{children}</>
  ),
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  TooltipContent: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  TooltipTrigger: ({ children }: { children: React.ReactNode }) => (
    <>{children}</>
  ),
}));

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {
    ARTIFACTS: "ARTIFACTS",
  },
  useGetFlag: () => mockArtifactsEnabled(),
}));

vi.mock("../../ChatMessagesContainer/ChatMessagesContainer", () => ({
  ChatMessagesContainer: ({
    bottomContentPadding,
  }: {
    bottomContentPadding?: number;
  }) => (
    <div
      data-testid="chat-messages-container"
      data-bottom-padding={bottomContentPadding ?? 0}
    />
  ),
}));

vi.mock("../../CopilotChatActionsProvider/CopilotChatActionsProvider", () => ({
  CopilotChatActionsProvider: ({ children }: { children: React.ReactNode }) => (
    <>{children}</>
  ),
}));

vi.mock("../../EmptySession/EmptySession", () => ({
  EmptySession: () => <div data-testid="empty-session" />,
}));

vi.mock("../../UsageLimits/useIsUsageLimitReached", () => ({
  useIsUsageLimitReached: () => mockIsUsageLimitReached(),
}));

vi.mock(
  "../../UsageLimits/UsageLimitReachedCard/UsageLimitReachedCard",
  () => ({
    UsageLimitReachedCard: () => <div role="alert">Usage limit reached</div>,
  }),
);

class MockResizeObserver {
  observe() {}
  disconnect() {}
  unobserve() {}
}

const baseProps = {
  messages: [] as never[],
  status: "ready",
  error: undefined,
  sessionId: "session-123",
  isLoadingSession: false,
  isSessionError: false,
  isCreatingSession: false,
  isReconnecting: false,
  isRestoringActiveSession: false,
  restoreStatusMessage: null,
  activeStreamStartedAt: null,
  isUserStopping: false,
  isSyncing: false,
  onCreateSession: vi.fn(),
  onSend: vi.fn(),
  onStop: vi.fn(),
  onEnqueue: vi.fn(),
  queuedMessages: [],
  isUploadingFiles: false,
  hasMoreMessages: false,
  isLoadingMore: false,
  onLoadMore: vi.fn(),
  droppedFiles: [],
  onDroppedFilesConsumed: vi.fn(),
  historicalDurations: new Map<string, number>(),
};

describe("ChatContainer", () => {
  beforeEach(() => {
    mockIsUsageLimitReached.mockReturnValue(false);
    mockArtifactsEnabled.mockReturnValue(false);
    mockShareState({ is_shared: false });
    resetCopilotStore();
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: clipboardWrite },
    });
    clipboardWrite.mockClear();
    vi.stubGlobal("ResizeObserver", MockResizeObserver);
  });

  afterEach(() => {
    cleanup();
    resetCopilotStore();
    vi.clearAllMocks();
    vi.unstubAllGlobals();
  });

  it("renders the blurred usage-limit backdrop only when the limit is reached", () => {
    mockIsUsageLimitReached.mockReturnValue(true);

    render(<ChatContainer {...baseProps} />);

    const backdrop = screen.getByTestId("usage-limit-backdrop");

    expect(screen.getByRole("alert")).toBeDefined();
    expect(backdrop.className).toContain("backdrop-blur-lg");
    expect(backdrop.className).toContain("[mask-image:linear-gradient");
    expect(backdrop.className).toContain("radial-gradient");
  });

  it("does not render the usage-limit backdrop while usage is still available", () => {
    render(<ChatContainer {...baseProps} />);

    expect(screen.queryByTestId("usage-limit-backdrop")).toBeNull();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("does not render the shared-chat notice for unshared chats", async () => {
    render(<ChatContainer {...baseProps} />);

    await waitFor(() => {
      expect(screen.queryByText("This chat is shared")).toBeNull();
    });
  });

  it("renders the shared-chat notice for chats the user has shared", async () => {
    mockShareState({ is_shared: true, share_token: SHARED_TOKEN });

    render(<ChatContainer {...baseProps} />);

    expect(await screen.findByText("This chat is shared")).toBeDefined();
    expect(screen.getByRole("button", { name: /copy link/i })).toBeDefined();
  });

  it("copies the shared-chat link from the owner notice", async () => {
    mockShareState({ is_shared: true, share_token: SHARED_TOKEN });

    render(<ChatContainer {...baseProps} />);

    fireEvent.click(await screen.findByRole("button", { name: /copy link/i }));

    await waitFor(() => {
      expect(clipboardWrite).toHaveBeenCalledTimes(1);
    });
    expect(clipboardWrite.mock.calls[0][0]).toMatch(
      new RegExp(`/share/chat/${SHARED_TOKEN}$`),
    );
    expect(
      await screen.findByRole("button", { name: /copied/i }),
    ).toBeDefined();
  });

  describe("auto-open artifact panel behavior", () => {
    it("does not auto-open the artifact panel on initial render", () => {
      mockArtifactsEnabled.mockReturnValue(true);

      render(<ChatContainer {...baseProps} />);

      expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
      const wrapper = screen.getByTestId(
        "chat-messages-container",
      ).parentElement;
      expect(wrapper?.className).toContain("max-w-3xl");
    });

    it("does not auto-open when rerendering within the same session", () => {
      mockArtifactsEnabled.mockReturnValue(true);

      const { rerender } = render(<ChatContainer {...baseProps} />);
      rerender(<ChatContainer {...baseProps} />);

      expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
      const wrapper = screen.getByTestId(
        "chat-messages-container",
      ).parentElement;
      expect(wrapper?.className).toContain("max-w-3xl");
    });

    it("clears the artifact preview when sessionId changes", () => {
      mockArtifactsEnabled.mockReturnValue(true);
      useCopilotUIStore
        .getState()
        .openArtifact(makeArtifact(ARTIFACT_A_ID, "a.txt"));
      useCopilotUIStore
        .getState()
        .openArtifact(makeArtifact(ARTIFACT_B_ID, "b.txt"));

      const { rerender } = render(
        <ChatContainer {...baseProps} sessionId="s1" />,
      );

      // The preview drawer is driven by activeArtifact, not `isOpen`.
      expect(
        useCopilotUIStore.getState().artifactPanel.activeArtifact?.id,
      ).toBe(ARTIFACT_B_ID);

      rerender(<ChatContainer {...baseProps} sessionId="s2" />);

      // `isOpen` is shared with ContextPanel and intentionally left alone
      // on session switches; the drawer hides via activeArtifact being null.
      const panel = useCopilotUIStore.getState().artifactPanel;
      expect(panel.activeArtifact).toBeNull();
      expect(panel.history).toEqual([]);
      const wrapper = screen.getByTestId(
        "chat-messages-container",
      ).parentElement;
      expect(wrapper?.className).toContain("max-w-3xl");
    });

    it("does not carry a stale back stack into the next session", () => {
      mockArtifactsEnabled.mockReturnValue(true);
      useCopilotUIStore
        .getState()
        .openArtifact(makeArtifact(ARTIFACT_A_ID, "a.txt"));
      useCopilotUIStore
        .getState()
        .openArtifact(makeArtifact(ARTIFACT_B_ID, "b.txt"));

      const { rerender } = render(
        <ChatContainer {...baseProps} sessionId="s1" />,
      );
      rerender(<ChatContainer {...baseProps} sessionId="s2" />);

      useCopilotUIStore.getState().openArtifact(makeArtifact("c", "c.txt"));

      const panel = useCopilotUIStore.getState().artifactPanel;
      expect(panel.activeArtifact?.id).toBe("c");
      expect(panel.history).toEqual([]);
    });

    it("clears artifact preview on unmount so nav-away cannot resurrect it (SECRT-2254)", () => {
      mockArtifactsEnabled.mockReturnValue(true);
      useCopilotUIStore
        .getState()
        .openArtifact(makeArtifact(ARTIFACT_A_ID, "a.txt"));
      expect(
        useCopilotUIStore.getState().artifactPanel.activeArtifact?.id,
      ).toBe(ARTIFACT_A_ID);

      const { unmount } = render(<ChatContainer {...baseProps} />);
      unmount();

      // The drawer's visibility is gated on activeArtifact, so clearing it
      // is what prevents a resurrection on nav-away. `isOpen` belongs to
      // ContextPanel and must not be flipped here.
      const panel = useCopilotUIStore.getState().artifactPanel;
      expect(panel.activeArtifact).toBeNull();
      expect(panel.history).toEqual([]);
    });

    it("does not re-open a panel whose store state is stale on fresh mount (SECRT-2220)", () => {
      mockArtifactsEnabled.mockReturnValue(true);
      useCopilotUIStore.setState({
        artifactPanel: {
          isOpen: true,
          activeArtifact: makeArtifact(ARTIFACT_A_ID, "stale.txt"),
          history: [],
          activeTab: "files",
        },
      });

      const { unmount } = render(<ChatContainer {...baseProps} />);
      unmount();

      render(<ChatContainer {...baseProps} />);

      // The drawer is gated on activeArtifact — clearing it is enough to
      // suppress the stale preview. `isOpen` is owned by ContextPanel and
      // is no longer flipped by the unmount cleanup.
      expect(
        useCopilotUIStore.getState().artifactPanel.activeArtifact,
      ).toBeNull();
      const wrapper = screen.getByTestId(
        "chat-messages-container",
      ).parentElement;
      expect(wrapper?.className).toContain("max-w-3xl");
    });
  });
});
