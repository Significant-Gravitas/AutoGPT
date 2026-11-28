import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { getCurrentUser, validateSession } from "@/lib/auth/actions";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { FIRST_RENDER_SESSION_DEADLINE_MS } from "@/lib/auth/server/getFirstRenderUser";
import { getServerSession } from "@/lib/auth/server/getServerSession";
import type { User } from "@/lib/auth/types";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import PlatformLayout from "../../layout";
import HomePage from "../page";

const routerPush = vi.hoisted(() => vi.fn());
const router = vi.hoisted(() => ({
  push: routerPush,
  replace: vi.fn(),
  refresh: vi.fn(),
  prefetch: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => router,
  usePathname: () => "/home",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

// The browser's session check: each call waits until the test answers it.
vi.mock("@/lib/auth/actions", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/auth/actions")>()),
  getCurrentUser: vi.fn(),
  validateSession: vi.fn(),
}));

function passThrough({ children }: { children?: ReactNode }) {
  return <>{children}</>;
}

// The layout's chrome and the chat itself aren't under test: only whether the
// page gets past its sign-in gate.
vi.mock("@/contexts/AutoPilotBridgeContext", () => ({
  AutoPilotBridgeProvider: passThrough,
}));
vi.mock("@/services/network-status/NetworkStatusMonitor", () => ({
  NetworkStatusMonitor: () => null,
}));
vi.mock("@/services/push-notifications/PushNotificationProvider", () => ({
  PushNotificationProvider: () => null,
}));
vi.mock("@/services/trials/TrialStatusSync", () => ({
  TrialStatusSync: () => null,
}));
vi.mock("@/services/pro-activation/ProActivationProvider", () => ({
  ProActivationProvider: passThrough,
}));
vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: passThrough,
}));
vi.mock("@/app/(platform)/PlatformChrome/PlatformChrome", () => ({
  PlatformChrome: passThrough,
}));
vi.mock("@/components/molecules/Sidebar/Sidebar", () => ({
  SidebarProvider: passThrough,
}));
vi.mock("@/app/(platform)/copilot/CopilotChatHost", () => ({
  CopilotChatHost: () => <div data-testid="chat" />,
}));
vi.mock(
  "@/app/(platform)/copilot/components/ContextPanel/ContextPanelAutoOpen",
  () => ({ ContextPanelAutoOpen: () => null }),
);
vi.mock(
  "@/app/(platform)/copilot/components/CopilotModals/CopilotModals",
  () => ({
    CopilotModals: () => null,
  }),
);
vi.mock(
  "@/app/(platform)/copilot/components/FileDropZone/FileDropZone",
  () => ({
    FileDropZone: passThrough,
  }),
);
vi.mock(
  "@/app/(platform)/copilot/components/NotificationBanner/NotificationBanner",
  () => ({ NotificationBanner: () => null }),
);
vi.mock(
  "@/components/layout/TopUpPrompt/LowCreditBanner/LowCreditBanner",
  () => ({ LowCreditBanner: () => null }),
);

const USER: User = {
  id: "user-1",
  email: "pat@example.com",
  role: "authenticated",
  user_metadata: { name: "Pat" },
};

// What Better Auth answers the layout for a request with a live session.
const SESSION = {
  user: {
    id: USER.id,
    email: USER.email,
    emailVerified: true,
    name: "Pat",
    role: "user",
    createdAt: new Date("2026-01-01T00:00:00Z"),
  },
  session: { id: "session-1", userId: USER.id },
};

const SIGNED_OUT = { user: null, isValid: false };

// Checks a test left unanswered, answered after it so they can't hold the
// store's one-check slot into the next test.
const unanswered: Array<() => void> = [];

function deferred<T>(answerAfterTest: T) {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  unanswered.push(() => resolve(answerAfterTest));
  return { promise, resolve };
}

async function homePage() {
  return render(await PlatformLayout({ children: <HomePage /> }));
}

describe("/home first paint", () => {
  beforeEach(() => {
    useAuthStore.getState().cleanup();
    useAuthStore.setState(useAuthStore.getInitialState(), true);
    routerPush.mockReset();
    vi.mocked(getServerSession).mockResolvedValue(null);
    vi.mocked(getCurrentUser).mockResolvedValue({ user: null });
    vi.mocked(validateSession).mockResolvedValue(SIGNED_OUT);
  });

  afterEach(async () => {
    vi.useRealTimers();
    for (const answer of unanswered.splice(0)) answer();
    await waitFor(() =>
      expect(useAuthStore.getState().isValidating).toBe(false),
    );
    vi.clearAllMocks();
  });

  test("with the session the server read, the page shows before the browser's session check answers", async () => {
    vi.mocked(getServerSession).mockResolvedValue(
      SESSION as unknown as Awaited<ReturnType<typeof getServerSession>>,
    );
    const check =
      deferred<Awaited<ReturnType<typeof validateSession>>>(SIGNED_OUT);
    vi.mocked(validateSession).mockReturnValue(check.promise);

    await homePage();

    expect(await screen.findByTestId("chat")).toBeDefined();
    expect(screen.queryByTestId("chat-frame")).toBeNull();
    // The check still runs, in the background.
    await waitFor(() => expect(validateSession).toHaveBeenCalledWith("/home"));
    expect(getCurrentUser).not.toHaveBeenCalled();

    check.resolve({ user: USER, isValid: true });
    await waitFor(() =>
      expect(useAuthStore.getState().isValidating).toBe(false),
    );
    expect(screen.getByTestId("chat")).toBeDefined();
  });

  test("a session that ended after the server read it is signed out by the browser's check", async () => {
    vi.mocked(getServerSession).mockResolvedValue(
      SESSION as unknown as Awaited<ReturnType<typeof getServerSession>>,
    );
    const check =
      deferred<Awaited<ReturnType<typeof validateSession>>>(SIGNED_OUT);
    vi.mocked(validateSession).mockReturnValue(check.promise);

    await homePage();
    expect(await screen.findByTestId("chat")).toBeDefined();
    await waitFor(() => expect(validateSession).toHaveBeenCalled());

    check.resolve({
      user: null,
      isValid: false,
      redirectPath: "/login?next=%2Fhome",
    });

    expect(await screen.findByTestId("chat-frame")).toBeDefined();
    expect(screen.queryByTestId("chat")).toBeNull();
    expect(routerPush).toHaveBeenCalledWith("/login?next=%2Fhome");
  });

  test("without a server session, the page waits for the browser's session check as before", async () => {
    const lookup = deferred<Awaited<ReturnType<typeof getCurrentUser>>>({
      user: null,
    });
    vi.mocked(getCurrentUser).mockReturnValue(lookup.promise);

    await homePage();

    await waitFor(() => expect(getCurrentUser).toHaveBeenCalled());
    expect(screen.getByTestId("chat-frame")).toBeDefined();
    expect(screen.queryByTestId("chat")).toBeNull();

    lookup.resolve({ user: USER });

    expect(await screen.findByTestId("chat")).toBeDefined();
  });

  test("a server session read that doesn't answer in time leaves the page to the browser's session check", async () => {
    vi.mocked(getServerSession).mockReturnValue(new Promise(() => {}));
    const lookup = deferred<Awaited<ReturnType<typeof getCurrentUser>>>({
      user: null,
    });
    vi.mocked(getCurrentUser).mockReturnValue(lookup.promise);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
      const layout = PlatformLayout({ children: <HomePage /> });
      await vi.advanceTimersByTimeAsync(FIRST_RENDER_SESSION_DEADLINE_MS);
      const page = await layout;
      vi.useRealTimers();
      render(page);

      await waitFor(() => expect(getCurrentUser).toHaveBeenCalled());
      expect(screen.getByTestId("loader")).toBeDefined();
      expect(screen.queryByTestId("chat")).toBeNull();

      lookup.resolve({ user: USER });

      expect(await screen.findByTestId("chat")).toBeDefined();
    } finally {
      vi.useRealTimers();
      warn.mockRestore();
    }
  });
});
