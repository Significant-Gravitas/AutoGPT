import { getGetV2ListSessionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import {
  afterAll,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import { useCopilotUIStore } from "../../../store";
import { MobileDrawer } from "../MobileDrawer";

vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: () => ({ isUserLoading: false, isLoggedIn: true }),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: () => true,
  };
});

vi.mock("@/components/molecules/Popover/Popover", () => {
  function Popover({ children }: { children: React.ReactNode }) {
    return <div>{children}</div>;
  }
  function PopoverTrigger({ children }: { children: React.ReactNode }) {
    return <div>{children}</div>;
  }
  function PopoverContent({ children }: { children: React.ReactNode }) {
    return <div>{children}</div>;
  }
  return { Popover, PopoverTrigger, PopoverContent };
});

vi.mock("../../UsageLimits/StorageBar", () => ({
  StorageBar: () => null,
}));

function mockUsageResponse() {
  const future = new Date(Date.now() + 3600 * 1000).toISOString();
  server.use(
    http.get("*/api/chat/usage", () =>
      HttpResponse.json({
        daily: { percent_used: 10, resets_at: future },
        weekly: { percent_used: 20, resets_at: future },
        tier: "BASIC",
      }),
    ),
  );
}

describe("MobileDrawer controls", () => {
  const originalSetPointerCapture = HTMLElement.prototype.setPointerCapture;
  const originalReleasePointerCapture =
    HTMLElement.prototype.releasePointerCapture;
  const originalHasPointerCapture = HTMLElement.prototype.hasPointerCapture;

  beforeAll(() => {
    HTMLElement.prototype.setPointerCapture = vi.fn();
    HTMLElement.prototype.releasePointerCapture = vi.fn();
    HTMLElement.prototype.hasPointerCapture = vi.fn(() => false);
  });

  afterAll(() => {
    HTMLElement.prototype.setPointerCapture = originalSetPointerCapture;
    HTMLElement.prototype.releasePointerCapture = originalReleasePointerCapture;
    HTMLElement.prototype.hasPointerCapture = originalHasPointerCapture;
  });

  beforeEach(() => {
    useCopilotUIStore.setState({
      isDrawerOpen: true,
      isSearchOpen: false,
      completedSessionIDs: new Set<string>(),
      isNotificationsEnabled: false,
      isSoundEnabled: true,
    });
    server.use(getGetV2ListSessionsMockHandler200({ sessions: [], total: 0 }));
    mockUsageResponse();
  });

  it("surfaces the usage and notification controls inside the drawer header", async () => {
    render(<MobileDrawer />);

    expect(
      await screen.findByRole("button", { name: /usage limits/i }),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: /notification settings/i }),
    ).toBeDefined();
  });

  it("does not render the legacy standalone sound toggle (folded into NotificationToggle)", async () => {
    render(<MobileDrawer />);

    // Wait for header to render so we don't false-pass before mount.
    await screen.findByRole("button", { name: /usage limits/i });

    expect(
      screen.queryByRole("button", {
        name: /enable notification sound|disable notification sound/i,
      }),
    ).toBeNull();
  });
});
