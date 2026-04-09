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

const mockUseSupabase = vi.hoisted(() => vi.fn());

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: mockUseSupabase,
}));

const testUser = {
  id: "user-1",
  email: "user@example.com",
  app_metadata: {},
  user_metadata: {},
  aud: "authenticated",
  created_at: "2026-01-01T00:00:00.000Z",
};

describe("SettingsPage", () => {
  beforeEach(() => {
    mockUseSupabase.mockReturnValue({
      user: testUser,
      isLoggedIn: true,
      isUserLoading: false,
      supabase: {},
    });
  });

  test("renders the account actions", async () => {
    server.use(
      getGetV1GetNotificationPreferencesMockHandler({
        user_id: "user-1",
        email: "user@example.com",
        preferences: {
          AGENT_RUN: true,
          ZERO_BALANCE: false,
          LOW_BALANCE: false,
          BLOCK_EXECUTION_FAILED: true,
          CONTINUOUS_AGENT_ERROR: false,
          DAILY_SUMMARY: false,
          WEEKLY_SUMMARY: true,
          MONTHLY_SUMMARY: false,
          AGENT_APPROVED: true,
          AGENT_REJECTED: true,
        },
        daily_limit: 0,
        emails_sent_today: 0,
        last_reset_date: new Date("2026-01-01T00:00:00.000Z"),
      }),
      getGetV1GetUserTimezoneMockHandler({ timezone: "Asia/Kolkata" }),
      getPostV1UpdateUserEmailMockHandler({}),
      getPostV1UpdateNotificationPreferencesMockHandler({
        user_id: "user-1",
        email: "user@example.com",
        preferences: {},
        daily_limit: 0,
        emails_sent_today: 0,
        last_reset_date: new Date("2026-01-01T00:00:00.000Z"),
      }),
    );

    render(<SettingsPage />);

    const emailInput = await screen.findByLabelText("Email");
    expect((emailInput as HTMLInputElement).value).toBe("user@example.com");
    expect(
      screen.getByRole("link", { name: "Reset password" }).getAttribute("href"),
    ).toBe("/reset-password");
  });

  test("saves notification preference changes", async () => {
    let submittedPreferences:
      | {
          email: string;
          preferences: Record<string, boolean>;
        }
      | undefined;

    server.use(
      getGetV1GetNotificationPreferencesMockHandler({
        user_id: "user-1",
        email: "user@example.com",
        preferences: {
          AGENT_RUN: false,
          ZERO_BALANCE: false,
          LOW_BALANCE: false,
          BLOCK_EXECUTION_FAILED: false,
          CONTINUOUS_AGENT_ERROR: false,
          DAILY_SUMMARY: false,
          WEEKLY_SUMMARY: false,
          MONTHLY_SUMMARY: false,
          AGENT_APPROVED: false,
          AGENT_REJECTED: false,
        },
        daily_limit: 0,
        emails_sent_today: 0,
        last_reset_date: new Date("2026-01-01T00:00:00.000Z"),
      }),
      getGetV1GetUserTimezoneMockHandler({ timezone: "Asia/Kolkata" }),
      getPostV1UpdateUserEmailMockHandler({}),
      getPostV1UpdateNotificationPreferencesMockHandler(async ({ request }) => {
        submittedPreferences = (await request.json()) as {
          email: string;
          preferences: Record<string, boolean>;
        };

        return {
          user_id: "user-1",
          email: submittedPreferences.email,
          preferences: submittedPreferences.preferences,
          daily_limit: 0,
          emails_sent_today: 0,
          last_reset_date: new Date("2026-01-01T00:00:00.000Z"),
        };
      }),
    );

    render(<SettingsPage />);

    fireEvent.click(
      await screen.findByRole("switch", { name: "Agent Run Notifications" }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Save preferences" }));

    await waitFor(() => {
      expect(submittedPreferences?.preferences.AGENT_RUN).toBe(true);
    });
  });
});
