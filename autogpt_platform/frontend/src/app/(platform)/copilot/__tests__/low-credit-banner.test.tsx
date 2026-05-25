import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import {
  getGetV1GetUserCreditsMockHandler,
  getGetV1GetAutoTopUpMockHandler,
} from "@/app/api/__generated__/endpoints/credits/credits.msw";
import { Key } from "@/services/storage/local-storage";
import { TopUpPromptProvider } from "@/components/layout/TopUpPrompt/TopUpPromptProvider";
import { CopilotPage } from "../CopilotPage";

// Stub the heavy chat subtree so the page mounts without websockets/streams.
// We only need a stable element to assert the page has rendered, plus the
// banner that TopUpPromptProvider + LowCreditBanner inject at the top.
vi.mock("../components/ChatContainer/ChatContainer", () => ({
  ChatContainer: () => <div data-testid="chat-container" />,
}));
vi.mock("../components/ChatSidebar/ChatSidebar", () => ({
  ChatSidebar: () => <div data-testid="chat-sidebar" />,
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

// Keep the chat host off the network — it reads useCopilotPage purely to
// render ChatContainer (which is stubbed above).
vi.mock("../useCopilotPage", () => ({
  useCopilotPage: () => ({
    sessionId: null,
    messages: [],
    status: "ready",
    error: undefined,
    stop: vi.fn(),
    isReconnecting: false,
    createSession: vi.fn(),
    onSend: vi.fn(),
    isLoadingSession: false,
    isSessionError: false,
    isCreatingSession: false,
    isUploadingFiles: false,
    hasMoreMessages: false,
    isLoadingMore: false,
    loadMore: vi.fn(),
    turnStats: new Map(),
    rateLimitMessage: null,
    dismissRateLimit: vi.fn(),
    sessionDryRun: false,
  }),
}));

vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: () => ({ isUserLoading: false, isLoggedIn: true }),
}));

// sessionId is read via nuqs to key the chat-host subtree; stub it.
vi.mock("nuqs", () => ({
  parseAsString: {},
  useQueryState: () => [null, vi.fn()],
}));

// Billing must be on for the provider to derive `isOutOfCredits`; keep the
// real `Flag` enum so the page's other flag reads (e.g. ARTIFACTS) resolve.
vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) => flag === actual.Flag.ENABLE_PLATFORM_PAYMENT,
  };
});

function setupCredits(credits: number) {
  server.use(
    getGetV1GetUserCreditsMockHandler({ credits }),
    getGetV1GetAutoTopUpMockHandler({ amount: 0, threshold: 0 }),
  );
}

function renderCopilotWithTopUpPrompt() {
  return render(
    <TopUpPromptProvider>
      <CopilotPage />
    </TopUpPromptProvider>,
  );
}

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  localStorage.clear();
});

describe("Autopilot low-credit banner", () => {
  test("shows the banner when the user is out of credits", async () => {
    // Suppress the daily auto-opener so only the banner is under test.
    localStorage.setItem(
      Key.TOP_UP_MODAL_LAST_SHOWN,
      new Date().toDateString(),
    );
    setupCredits(0);

    renderCopilotWithTopUpPrompt();

    expect(await screen.findByText(/out of automation credits/i)).toBeDefined();
  });

  test("hides the banner when the user still has credits", async () => {
    setupCredits(500);

    renderCopilotWithTopUpPrompt();

    // Let the chat host mount so the page has fully rendered before asserting absence.
    await screen.findByTestId("chat-container");

    expect(screen.queryByText(/out of automation credits/i)).toBeNull();
  });
});
