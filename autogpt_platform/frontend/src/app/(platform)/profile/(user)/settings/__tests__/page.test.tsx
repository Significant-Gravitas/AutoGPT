import type { ReactNode } from "react";
import {
  render,
  screen,
  fireEvent,
  waitFor,
} from "@/tests/integrations/test-utils";
import {
  getGetV1GetNotificationPreferencesMockHandler,
  getGetV1GetUserTimezoneMockHandler,
  getPostV1UpdateNotificationPreferencesMockHandler,
  getPostV1UpdateUserEmailMockHandler,
} from "@/app/api/__generated__/endpoints/auth/auth.msw";
import { server } from "@/mocks/mock-server";
import SettingsPage from "../page";
import { beforeEach, describe, expect, test, vi } from "vitest";

const mockUseAuth = vi.hoisted(() => vi.fn());

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: mockUseAuth,
}));

const testUser = {
  id: "user-1",
  email: "user@example.com",
  app_metadata: {},
  user_metadata: {},
  aud: "authenticated",
  created_at: "2026-01-01T00:00:00.000Z",
};

const defaultPreferences = {
  user_id: "user-1",
  email: "user@example.com",
  briefing_frequency: "WEEKLY" as const,
  alerts_enabled: true,
  store_verdicts_enabled: true,
  daily_limit: 0,
};

describe("SettingsPage", () => {
  beforeEach(() => {
    mockUseAuth.mockReturnValue({
      user: testUser,
      isLoggedIn: true,
      isUserLoading: false,
    });
  });

  test("renders the account actions", async () => {
    server.use(
      getGetV1GetNotificationPreferencesMockHandler(defaultPreferences),
      getGetV1GetUserTimezoneMockHandler({ timezone: "Asia/Kolkata" }),
      getPostV1UpdateUserEmailMockHandler({}),
      getPostV1UpdateNotificationPreferencesMockHandler(defaultPreferences),
    );

    render(<SettingsPage />);

    const emailInput = await screen.findByLabelText("Email");
    expect((emailInput as HTMLInputElement).value).toBe("user@example.com");
    expect(
      screen.getByRole("link", { name: "Reset password" }).getAttribute("href"),
    ).toBe("/reset-password");
  });

  test("saves notification preference changes", async () => {
    let submitted:
      | {
          email: string;
          briefing_frequency: string;
          alerts_enabled: boolean;
          store_verdicts_enabled: boolean;
        }
      | undefined;

    server.use(
      getGetV1GetNotificationPreferencesMockHandler(defaultPreferences),
      getGetV1GetUserTimezoneMockHandler({ timezone: "Asia/Kolkata" }),
      getPostV1UpdateUserEmailMockHandler({}),
      getPostV1UpdateNotificationPreferencesMockHandler(async ({ request }) => {
        submitted = (await request.json()) as typeof submitted;
        return { ...defaultPreferences, ...submitted };
      }),
    );

    render(<SettingsPage />);

    fireEvent.click(await screen.findByRole("switch", { name: "Alerts" }));
    fireEvent.click(screen.getByRole("button", { name: "Save preferences" }));

    await waitFor(() => {
      expect(submitted?.alerts_enabled).toBe(false);
    });
  });
});
