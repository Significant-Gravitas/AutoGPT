import React from "react";
import { render, screen, cleanup } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ChatContainer } from "../ChatContainer";

const mockIsUsageLimitReached = vi.fn();
const mockArtifactsEnabled = vi.fn(() => false);
const mockArtifactPanelOpen = vi.fn(() => false);

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

vi.mock("../../../store", () => ({
  useCopilotUIStore: (
    selector?: (state: { artifactPanel: { isOpen: boolean } }) => unknown,
  ) => {
    const state = { artifactPanel: { isOpen: mockArtifactPanelOpen() } };
    return typeof selector === "function" ? selector(state) : state;
  },
}));

vi.mock("../useAutoOpenArtifacts", () => ({
  useAutoOpenArtifacts: vi.fn(),
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

vi.mock("../../UsageLimits/UsageLimitReachedCard", () => ({
  useIsUsageLimitReached: () => mockIsUsageLimitReached(),
  UsageLimitReachedCard: () => <div role="alert">Usage limit reached</div>,
}));

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
    mockArtifactPanelOpen.mockReturnValue(false);
    vi.stubGlobal("ResizeObserver", MockResizeObserver);
  });

  afterEach(() => {
    cleanup();
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
});
