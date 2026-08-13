import type { ReactNode } from "react";
import { beforeEach, describe, expect, test, vi } from "vitest";

import {
  getGetV1GetNotificationPreferencesMockHandler,
  getGetV1GetUserTimezoneMockHandler,
  getPostV1UpdateNotificationPreferencesMockHandler,
  getPostV1UpdateUserEmailMockHandler,
  getPostV1UpdateUserTimezoneMockHandler,
} from "@/app/api/__generated__/endpoints/auth/auth.msw";
import type { NotificationPreference } from "@/app/api/__generated__/models/notificationPreference";
import type { NotificationPreferenceDTO } from "@/app/api/__generated__/models/notificationPreferenceDTO";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";

import SettingsPreferencesPage from "../page";

const mockUseAuth = vi.hoisted(() => vi.fn());
const mockUseGetFlag = vi.hoisted(() => vi.fn());
const mockUseSearchParams = vi.hoisted(() =>
  vi.fn(() => new URLSearchParams()),
);

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: mockUseAuth,
}));

vi.mock("next/navigation", async (importOriginal) => {
  const actual = await importOriginal<typeof import("next/navigation")>();
  return {
    ...actual,
    useSearchParams: mockUseSearchParams,
  };
});

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: mockUseGetFlag,
  };
});

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
} satisfies NotificationPreference;

function setupBaseHandlers(
  options: {
    timezone?: string;
    preferences?: Partial<typeof defaultPreferences>;
  } = {},
) {
  server.use(
    getGetV1GetNotificationPreferencesMockHandler({
      ...defaultPreferences,
      ...options.preferences,
    }),
    getGetV1GetUserTimezoneMockHandler({
      timezone: options.timezone ?? "Asia/Kolkata",
    }),
    getPostV1UpdateUserEmailMockHandler({}),
    getPostV1UpdateNotificationPreferencesMockHandler(defaultPreferences),
    getPostV1UpdateUserTimezoneMockHandler({
      timezone: "Europe/London",
    }),
  );
}

describe("SettingsPreferencesPage", () => {
  beforeEach(() => {
    mockUseAuth.mockReturnValue({
      user: testUser,
      isLoggedIn: true,
      isUserLoading: false,
    });
    // Notifications are behind settings-notifications; default the suite to
    // "on" so the notification cases exercise the card, and flip it off in
    // the case that asserts it stays hidden.
    mockUseGetFlag.mockReturnValue(true);
    mockUseSearchParams.mockReturnValue(new URLSearchParams());
  });

  test("renders Account card with current email and reset password link", async () => {
    mockUseGetFlag.mockReturnValue(false);
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

  test("switching Alerts off enables Save and persists the volume knob", async () => {
    let submitted: NotificationPreferenceDTO | undefined;

    server.use(
      getGetV1GetNotificationPreferencesMockHandler(defaultPreferences),
      getGetV1GetUserTimezoneMockHandler({ timezone: "Asia/Kolkata" }),
      getPostV1UpdateUserEmailMockHandler({}),
      getPostV1UpdateUserTimezoneMockHandler({ timezone: "Asia/Kolkata" }),
      getPostV1UpdateNotificationPreferencesMockHandler(async ({ request }) => {
        submitted = (await request.json()) as NotificationPreferenceDTO;
        return { ...defaultPreferences, ...submitted };
      }),
    );

    render(<SettingsPreferencesPage />);

    const saveButton = await screen.findByRole("button", {
      name: "Save changes",
    });
    expect((saveButton as HTMLButtonElement).disabled).toBe(true);

    fireEvent.click(await screen.findByRole("switch", { name: "Alerts" }));

    await waitFor(() => {
      expect((saveButton as HTMLButtonElement).disabled).toBe(false);
    });

    fireEvent.click(saveButton);

    await waitFor(() => {
      expect(submitted).toBeDefined();
    });
    expect(submitted!.alerts_enabled).toBe(false);
    // Billing and account messages are service mail and aren't represented
    // here at all, so there is nothing to switch off for them.
    expect(submitted!.briefing_frequency).toBe("WEEKLY");
  });

  test("applies an Alerts-only email footer link and saves it", async () => {
    let submitted: NotificationPreferenceDTO | undefined;
    mockUseSearchParams.mockReturnValue(new URLSearchParams({ f: "alerts" }));

    server.use(
      getGetV1GetNotificationPreferencesMockHandler({
        ...defaultPreferences,
        alerts_enabled: false,
      }),
      getGetV1GetUserTimezoneMockHandler({ timezone: "Asia/Kolkata" }),
      getPostV1UpdateUserEmailMockHandler({}),
      getPostV1UpdateUserTimezoneMockHandler({ timezone: "Asia/Kolkata" }),
      getPostV1UpdateNotificationPreferencesMockHandler(async ({ request }) => {
        submitted = (await request.json()) as NotificationPreferenceDTO;
        return { ...defaultPreferences, ...submitted };
      }),
    );

    render(<SettingsPreferencesPage />);

    const alertsSwitch = await screen.findByRole("switch", { name: "Alerts" });
    const saveButton = screen.getByRole("button", { name: "Save changes" });

    expect(alertsSwitch.getAttribute("aria-checked")).toBe("true");
    expect((saveButton as HTMLButtonElement).disabled).toBe(false);

    fireEvent.click(saveButton);

    await waitFor(() => {
      expect(submitted).toMatchObject({
        briefing_frequency: "OFF",
        alerts_enabled: true,
      });
    });
  });

  test("saves briefing frequency and marketplace review changes", async () => {
    let submitted: NotificationPreferenceDTO | undefined;

    server.use(
      getGetV1GetNotificationPreferencesMockHandler(defaultPreferences),
      getGetV1GetUserTimezoneMockHandler({ timezone: "Asia/Kolkata" }),
      getPostV1UpdateUserEmailMockHandler({}),
      getPostV1UpdateUserTimezoneMockHandler({ timezone: "Asia/Kolkata" }),
      getPostV1UpdateNotificationPreferencesMockHandler(async ({ request }) => {
        submitted = (await request.json()) as NotificationPreferenceDTO;
        return { ...defaultPreferences, ...submitted };
      }),
    );

    render(<SettingsPreferencesPage />);

    fireEvent.click(await screen.findByRole("combobox", { name: "Briefing" }));
    fireEvent.click(await screen.findByRole("option", { name: "Monthly" }));
    fireEvent.click(
      screen.getByRole("switch", { name: "Marketplace reviews" }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }));

    await waitFor(() => {
      expect(submitted).toMatchObject({
        briefing_frequency: "MONTHLY",
        alerts_enabled: true,
        store_verdicts_enabled: false,
      });
    });
  });

  test("Discard reverts an unsaved notification change", async () => {
    setupBaseHandlers();

    render(<SettingsPreferencesPage />);

    const saveButton = await screen.findByRole("button", {
      name: "Save changes",
    });
    const discardButton = screen.getByRole("button", { name: "Discard" });

    const alertsSwitch = await screen.findByRole("switch", { name: "Alerts" });
    const initialChecked = alertsSwitch.getAttribute("aria-checked");

    fireEvent.click(alertsSwitch);

    await waitFor(() => {
      expect((saveButton as HTMLButtonElement).disabled).toBe(false);
      expect((discardButton as HTMLButtonElement).disabled).toBe(false);
    });

    fireEvent.click(discardButton);

    await waitFor(() => {
      expect((saveButton as HTMLButtonElement).disabled).toBe(true);
      expect(alertsSwitch.getAttribute("aria-checked")).toBe(initialChecked);
    });
  });

  test("saving a timezone change posts the new value", async () => {
    let submittedTimezone: string | undefined;

    server.use(
      getGetV1GetNotificationPreferencesMockHandler(defaultPreferences),
      getGetV1GetUserTimezoneMockHandler({ timezone: "Asia/Kolkata" }),
      getPostV1UpdateUserEmailMockHandler({}),
      getPostV1UpdateNotificationPreferencesMockHandler(defaultPreferences),
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
      getGetV1GetNotificationPreferencesMockHandler(defaultPreferences),
      getGetV1GetUserTimezoneMockHandler({ timezone: "not-set" }),
      getPostV1UpdateUserEmailMockHandler({}),
      getPostV1UpdateNotificationPreferencesMockHandler(defaultPreferences),
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
  test("hides the notifications card when settings-notifications is off", async () => {
    mockUseGetFlag.mockReturnValue(false);
    setupBaseHandlers();

    render(<SettingsPreferencesPage />);

    expect(await screen.findByText("Time zone")).toBeDefined();
    expect(screen.queryByRole("combobox", { name: "Briefing" })).toBeNull();
    expect(screen.queryAllByRole("switch")).toHaveLength(0);
  });
});
