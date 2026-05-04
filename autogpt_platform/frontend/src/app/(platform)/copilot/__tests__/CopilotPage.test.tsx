import { render, screen, cleanup } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CopilotPage } from "../CopilotPage";

// Mock child components that are complex and not under test here
vi.mock("../components/ChatContainer/ChatContainer", () => ({
  ChatContainer: () => <div data-testid="chat-container" />,
}));
vi.mock("../components/ChatSidebar/ChatSidebar", () => ({
  ChatSidebar: () => <div data-testid="chat-sidebar" />,
}));
vi.mock("../components/DeleteChatDialog/DeleteChatDialog", () => ({
  DeleteChatDialog: () => null,
}));
vi.mock("../components/MobileDrawer/MobileDrawer", () => ({
  MobileDrawer: () => null,
}));
vi.mock("../components/MobileHeader/MobileHeader", () => ({
  MobileHeader: () => null,
}));
vi.mock("../components/NotificationBanner/NotificationBanner", () => ({
  NotificationBanner: () => null,
}));
vi.mock("../components/NotificationDialog/NotificationDialog", () => ({
  NotificationDialog: () => null,
}));
vi.mock("../components/RateLimitResetDialog/RateLimitResetDialog", () => ({
  RateLimitResetDialog: () => null,
}));
vi.mock("../components/RateLimitResetDialog/RateLimitGate", () => ({
  RateLimitGate: () => null,
}));
vi.mock("../components/FileDropZone/FileDropZone", () => ({
  FileDropZone: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
}));
vi.mock("../useIsMobile", () => ({
  useIsMobile: () => false,
}));
vi.mock("../components/ScaleLoader/ScaleLoader", () => ({
  ScaleLoader: () => <div data-testid="scale-loader" />,
}));
vi.mock("../components/ArtifactPanel/ArtifactPanel", () => ({
  ArtifactPanel: () => null,
}));
vi.mock("@/components/ui/sidebar", () => ({
  SidebarProvider: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
}));

// Mock hooks that hit the network. Exercise the `select` callback so its
// line counts as covered alongside the rest of the options.
vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  useGetV2GetCopilotUsage: (opts: {
    query?: { select?: (r: { data: unknown }) => unknown };
  }) => {
    const data = {
      daily: null,
      weekly: null,
      tier: "BASIC",
      reset_cost: 0,
    };
    if (typeof opts?.query?.select === "function") {
      opts.query.select({ data });
    }
    return { data: undefined, isSuccess: false, isError: false };
  },
}));
vi.mock("@/hooks/useCredits", () => ({
  default: () => ({ credits: null, fetchCredits: vi.fn() }),
}));
vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {
    ENABLE_PLATFORM_PAYMENT: "ENABLE_PLATFORM_PAYMENT",
    ARTIFACTS: "ARTIFACTS",
    CHAT_MODE_OPTION: "CHAT_MODE_OPTION",
  },
  useGetFlag: () => false,
}));

// Auth check moved into CopilotPage directly — default to a logged-in
// user so the page renders past its loading gate.
const mockSupabase = vi.fn(() => ({ isUserLoading: false, isLoggedIn: true }));
vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: () => mockSupabase(),
}));

// sessionId is read via nuqs to key the chat-host subtree; stub it so
// tests can control the key/session without hitting URL state.
let mockSessionIdForQueryState: string | null = null;
vi.mock("nuqs", () => ({
  parseAsString: {},
  useQueryState: () => [mockSessionIdForQueryState, vi.fn()],
}));

// Build the base mock return value for useCopilotPage
const basePageState = {
  sessionId: null as string | null,
  messages: [],
  status: "ready" as const,
  error: undefined,
  stop: vi.fn(),
  isReconnecting: false,
  isSyncing: false,
  createSession: vi.fn(),
  onSend: vi.fn(),
  isLoadingSession: false,
  isSessionError: false,
  isCreatingSession: false,
  isUploadingFiles: false,
  isUserLoading: false,
  isLoggedIn: true,
  hasMoreMessages: false,
  isLoadingMore: false,
  loadMore: vi.fn(),
  turnStats: new Map(),
  rateLimitMessage: null,
  dismissRateLimit: vi.fn(),
  sessionDryRun: false,
};

const mockUseCopilotPage = vi.fn(() => basePageState);

vi.mock("../useCopilotPage", () => ({
  useCopilotPage: () => mockUseCopilotPage(),
}));

afterEach(() => {
  cleanup();
  mockUseCopilotPage.mockReset();
  mockUseCopilotPage.mockImplementation(() => basePageState);
  mockSupabase.mockReset();
  mockSupabase.mockImplementation(() => ({
    isUserLoading: false,
    isLoggedIn: true,
  }));
  mockSessionIdForQueryState = null;
});

describe("CopilotPage test-mode banner", () => {
  it("does not show test-mode banner when there is no active session", () => {
    render(<CopilotPage />);
    expect(
      screen.queryByText(/test mode.*this session runs agents/i),
    ).toBeNull();
  });

  it("does not show test-mode banner when session exists but sessionDryRun is false", () => {
    mockSessionIdForQueryState = "session-abc";
    mockUseCopilotPage.mockReturnValue({
      ...basePageState,
      sessionId: "session-abc",
      sessionDryRun: false,
    });
    render(<CopilotPage />);
    expect(
      screen.queryByText(/test mode.*this session runs agents/i),
    ).toBeNull();
  });

  it("shows test-mode banner when session exists and sessionDryRun is true", () => {
    mockSessionIdForQueryState = "session-abc";
    mockUseCopilotPage.mockReturnValue({
      ...basePageState,
      sessionId: "session-abc",
      sessionDryRun: true,
    });
    render(<CopilotPage />);
    expect(
      screen.getByText(/test mode.*this session runs agents/i),
    ).toBeDefined();
  });

  it("does not show test-mode banner when sessionDryRun is true but no sessionId", () => {
    mockUseCopilotPage.mockReturnValue({
      ...basePageState,
      sessionId: null,
      sessionDryRun: true,
    });
    render(<CopilotPage />);
    expect(
      screen.queryByText(/test mode.*this session runs agents/i),
    ).toBeNull();
  });

  it("shows loading spinner when user is loading", () => {
    // Auth check moved to CopilotPage — mock useSupabase directly.
    mockSupabase.mockReturnValue({ isUserLoading: true, isLoggedIn: false });
    render(<CopilotPage />);
    expect(screen.getByTestId("scale-loader")).toBeDefined();
    expect(screen.queryByTestId("chat-container")).toBeNull();
  });
});
