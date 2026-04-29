import type { ReactNode } from "react";
import { beforeEach, describe, expect, test, vi } from "vitest";

import {
  getGetV1GetNotificationPreferencesMockHandler,
  getGetV1GetUserTimezoneMockHandler,
  getPostV1UpdateNotificationPreferencesMockHandler,
  getPostV1UpdateUserEmailMockHandler,
  getPostV1UpdateUserTimezoneMockHandler,
} from "@/app/api/__generated__/endpoints/auth/auth.msw";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";

import SettingsPreferencesPage from "../page";

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

const baseDate = new Date("2026-01-01T00:00:00.000Z");

const allFalsePreferences = {
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
};

function setupBaseHandlers(
  options: { timezone?: string; preferences?: Record<string, boolean> } = {},
) {
  server.use(
    getGetV1GetNotificationPreferencesMockHandler({
      user_id: "user-1",
      email: "user@example.com",
      preferences: options.preferences ?? allFalsePreferences,
      daily_limit: 0,
      emails_sent_today: 0,
      last_reset_date: baseDate,
    }),
    getGetV1GetUserTimezoneMockHandler({
      timezone: options.timezone ?? "Asia/Kolkata",
    }),
    getPostV1UpdateUserEmailMockHandler({}),
    getPostV1UpdateNotificationPreferencesMockHandler({
      user_id: "user-1",
      email: "user@example.com",
      preferences: {},
      daily_limit: 0,
      emails_sent_today: 0,
      last_reset_date: baseDate,
    }),
    getPostV1UpdateUserTimezoneMockHandler({
      timezone: "Europe/London",
    }),
  );
}

describe("SettingsPreferencesPage", () => {
  beforeEach(() => {
    mockUseSupabase.mockReturnValue({
      user: testUser,
      isLoggedIn: true,
      isUserLoading: false,
      supabase: {},
    });
  });

  test("renders Account card with current email and reset password link", async () => {
    setupBaseHandlers();

    render(<SettingsPreferencesPage />);

    expect(await screen.findByText("Account")).toBeDefined();
    expect(screen.getByText("user@example.com")).toBeDefined();
    expect(screen.getByText("Email")).toBeDefined();
    expect(screen.getByText("Password")).toBeDefined();

    const resetLink = screen.getByRole("link", { name: "Reset password" });
    expect(resetLink.getAttribute("href")).toBe("/reset-password");
  });

  test("opens email update dialog and closes on Cancel", async () => {
    setupBaseHandlers();

    render(<SettingsPreferencesPage />);

    const editButton = await screen.findByRole("button", {
      name: "Edit email",
    });
    fireEvent.click(editButton);

    const dialogInput = await screen.findByLabelText("Email");
    expect((dialogInput as HTMLInputElement).value).toBe("user@example.com");

    const updateButton = screen.getByRole("button", { name: "Update email" });
    expect((updateButton as HTMLButtonElement).disabled).toBe(true);

    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));

    await waitFor(() => {
      expect(screen.queryByLabelText("Email")).toBeNull();
    });
  });

  test("renders Time zone card with info tooltip trigger", async () => {
    setupBaseHandlers({ timezone: "Asia/Kolkata" });

    render(<SettingsPreferencesPage />);

    expect(await screen.findByText("Time zone")).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Time zone info" }),
    ).toBeDefined();
  });

  test("Save and Discard buttons start disabled when nothing has changed", async () => {
    setupBaseHandlers();

    render(<SettingsPreferencesPage />);

    const saveButton = await screen.findByRole("button", {
      name: "Save changes",
    });
    const discardButton = screen.getByRole("button", { name: "Discard" });

    expect((saveButton as HTMLButtonElement).disabled).toBe(true);
    expect((discardButton as HTMLButtonElement).disabled).toBe(true);
  });

  test("toggling a notification enables Save and persists on click", async () => {
    let submittedPreferences:
      | { email: string; preferences: Record<string, boolean> }
      | undefined;

    server.use(
      getGetV1GetNotificationPreferencesMockHandler({
        user_id: "user-1",
        email: "user@example.com",
        preferences: allFalsePreferences,
        daily_limit: 0,
        emails_sent_today: 0,
        last_reset_date: baseDate,
      }),
      getGetV1GetUserTimezoneMockHandler({ timezone: "Asia/Kolkata" }),
      getPostV1UpdateUserEmailMockHandler({}),
      getPostV1UpdateUserTimezoneMockHandler({ timezone: "Asia/Kolkata" }),
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
          last_reset_date: baseDate,
        };
      }),
    );

    render(<SettingsPreferencesPage />);

    const saveButton = await screen.findByRole("button", {
      name: "Save changes",
    });
    expect((saveButton as HTMLButtonElement).disabled).toBe(true);

    const switches = await screen.findAllByRole("switch");
    fireEvent.click(switches[0]);

    await waitFor(() => {
      expect((saveButton as HTMLButtonElement).disabled).toBe(false);
    });

    fireEvent.click(saveButton);

    await waitFor(() => {
      expect(submittedPreferences).toBeDefined();
    });
    const flippedKey = Object.entries(submittedPreferences!.preferences).find(
      ([, v]) => v === true,
    )?.[0];
    expect(flippedKey).toBeDefined();
  });

  test("saving a timezone change posts the new value", async () => {
    let submittedTimezone: string | undefined;

    server.use(
      getGetV1GetNotificationPreferencesMockHandler({
        user_id: "user-1",
        email: "user@example.com",
        preferences: allFalsePreferences,
        daily_limit: 0,
        emails_sent_today: 0,
        last_reset_date: baseDate,
      }),
      getGetV1GetUserTimezoneMockHandler({ timezone: "Asia/Kolkata" }),
      getPostV1UpdateUserEmailMockHandler({}),
      getPostV1UpdateNotificationPreferencesMockHandler({
        user_id: "user-1",
        email: "user@example.com",
        preferences: {},
        daily_limit: 0,
        emails_sent_today: 0,
        last_reset_date: baseDate,
      }),
      getPostV1UpdateUserTimezoneMockHandler(async ({ request }) => {
        const body = (await request.json()) as { timezone: string };
        submittedTimezone = body.timezone;
        return { timezone: body.timezone };
      }),
    );

    render(<SettingsPreferencesPage />);

    const select = await screen.findByRole("combobox", { name: "Timezone" });
    fireEvent.click(select);

    const option = await screen.findByRole("option", {
      name: /London/i,
    });
    fireEvent.click(option);

    const saveButton = screen.getByRole("button", { name: "Save changes" });

    await waitFor(() => {
      expect((saveButton as HTMLButtonElement).disabled).toBe(false);
    });

    fireEvent.click(saveButton);

    await waitFor(() => {
      expect(submittedTimezone).toBe("Europe/London");
    });
  });

  test("Save is enabled on first paint when server timezone is not-set, and saves the detected browser tz", async () => {
    const STUBBED_BROWSER_TZ = "America/New_York";
    const resolvedOptionsSpy = vi
      .spyOn(Intl.DateTimeFormat.prototype, "resolvedOptions")
      .mockReturnValue({
        timeZone: STUBBED_BROWSER_TZ,
      } as Intl.ResolvedDateTimeFormatOptions);

    let submittedTimezone: string | undefined;

    server.use(
      getGetV1GetNotificationPreferencesMockHandler({
        user_id: "user-1",
        email: "user@example.com",
        preferences: allFalsePreferences,
        daily_limit: 0,
        emails_sent_today: 0,
        last_reset_date: baseDate,
      }),
      getGetV1GetUserTimezoneMockHandler({ timezone: "not-set" }),
      getPostV1UpdateUserEmailMockHandler({}),
      getPostV1UpdateNotificationPreferencesMockHandler({
        user_id: "user-1",
        email: "user@example.com",
        preferences: {},
        daily_limit: 0,
        emails_sent_today: 0,
        last_reset_date: baseDate,
      }),
      getPostV1UpdateUserTimezoneMockHandler(async ({ request }) => {
        const body = (await request.json()) as { timezone: string };
        submittedTimezone = body.timezone;
        return { timezone: body.timezone };
      }),
    );

    try {
      render(<SettingsPreferencesPage />);

      const saveButton = await screen.findByRole("button", {
        name: "Save changes",
      });

      await waitFor(() => {
        expect((saveButton as HTMLButtonElement).disabled).toBe(false);
      });

      fireEvent.click(saveButton);

      await waitFor(() => {
        expect(submittedTimezone).toBe(STUBBED_BROWSER_TZ);
      });

      await waitFor(() => {
        expect((saveButton as HTMLButtonElement).disabled).toBe(true);
      });
    } finally {
      resolvedOptionsSpy.mockRestore();
    }
  });

  test("submitting a new email closes the dialog and calls the update endpoint", async () => {
    const fetchMock = vi.fn(async () =>
      Promise.resolve(
        new Response(JSON.stringify({ ok: true }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    setupBaseHandlers();

    render(<SettingsPreferencesPage />);

    fireEvent.click(await screen.findByRole("button", { name: "Edit email" }));

    const dialogInput = (await screen.findByLabelText(
      "Email",
    )) as HTMLInputElement;
    fireEvent.change(dialogInput, { target: { value: "new@example.com" } });

    const updateButton = screen.getByRole("button", { name: "Update email" });
    await waitFor(() => {
      expect((updateButton as HTMLButtonElement).disabled).toBe(false);
    });

    fireEvent.click(updateButton);

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/auth/user",
        expect.objectContaining({ method: "PUT" }),
      );
    });

    await waitFor(() => {
      expect(screen.queryByLabelText("Email")).toBeNull();
    });

    vi.unstubAllGlobals();
  });

  test("Discard reverts unsaved notification toggles", async () => {
    setupBaseHandlers();

    render(<SettingsPreferencesPage />);

    const saveButton = await screen.findByRole("button", {
      name: "Save changes",
    });
    const discardButton = screen.getByRole("button", { name: "Discard" });

    const switches = await screen.findAllByRole("switch");
    const targetSwitch = switches[0] as HTMLInputElement;
    const initialChecked = targetSwitch.getAttribute("aria-checked");

    fireEvent.click(targetSwitch);

    await waitFor(() => {
      expect((saveButton as HTMLButtonElement).disabled).toBe(false);
      expect((discardButton as HTMLButtonElement).disabled).toBe(false);
    });

    fireEvent.click(discardButton);

    await waitFor(() => {
      expect((saveButton as HTMLButtonElement).disabled).toBe(true);
      expect(targetSwitch.getAttribute("aria-checked")).toBe(initialChecked);
    });
  });
});
