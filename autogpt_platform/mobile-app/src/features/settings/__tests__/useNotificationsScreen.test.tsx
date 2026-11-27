import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react-native";
import type { ReactNode } from "react";

import {
  getNotificationPreferences,
  getTimezone,
  preferenceKeys,
  updateNotificationPreferences,
  type NotificationPreferences,
} from "@/lib/api/preferences";
import { sessionGeneration } from "@/lib/session";

import { useNotificationsScreen } from "../useNotificationsScreen";

jest.mock("expo-router", () => ({ router: {} }));
jest.mock("@/lib/session", () => ({ sessionGeneration: jest.fn() }));
jest.mock("@/lib/api/preferences", () => ({
  ...jest.requireActual("@/lib/api/preferences"),
  getNotificationPreferences: jest.fn(),
  getTimezone: jest.fn(),
  updateNotificationPreferences: jest.fn(),
}));

const saved: NotificationPreferences = {
  email: "learner@example.com",
  briefing_frequency: "WEEKLY",
  alerts_enabled: true,
  store_verdicts_enabled: false,
  daily_limit: 3,
};

let client: QueryClient;

function Wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

beforeEach(() => {
  client = new QueryClient({
    defaultOptions: {
      queries: { gcTime: Infinity },
      mutations: { gcTime: Infinity },
    },
  });
  jest.mocked(sessionGeneration).mockReturnValue(1);
  jest.mocked(getNotificationPreferences).mockResolvedValue(saved);
  jest.mocked(getTimezone).mockResolvedValue({ timezone: "Europe/Madrid" });
});

afterEach(() => {
  client.clear();
  jest.clearAllMocks();
});

test("settings writes stay ordered across screen visits", async () => {
  const daily = { ...saved, briefing_frequency: "DAILY" as const };
  let finish = () => {};
  jest
    .mocked(updateNotificationPreferences)
    .mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finish = () => resolve(daily);
        }),
    )
    .mockImplementationOnce(async (preferences) => preferences);
  const first = await renderHook(useNotificationsScreen, { wrapper: Wrapper });
  await waitFor(() => expect(first.result.current.preferences).toEqual(saved));
  await act(() => first.result.current.chooseBriefing("DAILY"));
  await first.unmount();

  // These writes model returning to a drawer screen while its earlier save runs.
  jest.mocked(getNotificationPreferences).mockResolvedValue(daily);
  const second = await renderHook(useNotificationsScreen, { wrapper: Wrapper });
  expect(second.result.current.isSavingPreferences).toBe(true);
  await act(() => second.result.current.setAlerts(false));
  expect(updateNotificationPreferences).toHaveBeenCalledTimes(1);

  await act(async () => finish());
  await waitFor(() =>
    expect(second.result.current.isSavingPreferences).toBe(false),
  );
  await act(() => second.result.current.setAlerts(false));
  expect(updateNotificationPreferences).toHaveBeenLastCalledWith({
    ...daily,
    alerts_enabled: false,
  });
});

test("a response after sign-out leaves the current account's preferences alone", async () => {
  let finish = () => {};
  jest.mocked(updateNotificationPreferences).mockImplementationOnce(
    (preferences) =>
      new Promise((resolve) => {
        finish = () => resolve(preferences);
      }),
  );
  const screen = await renderHook(useNotificationsScreen, { wrapper: Wrapper });
  await waitFor(() => expect(screen.result.current.preferences).toEqual(saved));
  await act(() => screen.result.current.chooseBriefing("DAILY"));
  await screen.unmount();

  const newAccount = { ...saved, email: "other@example.com" };
  jest.mocked(sessionGeneration).mockReturnValue(2);
  client.clear();
  client.setQueryData(preferenceKeys.notifications, newAccount);
  await act(async () => finish());

  expect(client.getQueryData(preferenceKeys.notifications)).toEqual(newAccount);
});
