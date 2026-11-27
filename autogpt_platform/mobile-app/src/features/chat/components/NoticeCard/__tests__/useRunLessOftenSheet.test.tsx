import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react-native";
import type { ReactNode } from "react";

import { getCadence, setCadence } from "@/lib/api/scheduledChats";

import { useRunLessOftenSheet } from "../components/useRunLessOftenSheet";

jest.mock("@/lib/session", () => ({ sessionGeneration: () => 1 }));
jest.mock("@/lib/api/scheduledChats", () => ({
  ...jest.requireActual("@/lib/api/scheduledChats"),
  getCadence: jest.fn(),
  setCadence: jest.fn(),
}));

test("selecting another cadence cannot reset a pending save", async () => {
  const client = new QueryClient({
    defaultOptions: {
      queries: { gcTime: Infinity },
      mutations: { gcTime: Infinity },
    },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  }
  const cadence = {
    schedule_key: "routine",
    title: "A routine",
    current: "Daily",
    timezone: "UTC",
    options: [
      { cron: "weekly", label: "Weekly" },
      { cron: "monthly", label: "Monthly" },
    ],
  };
  jest.mocked(getCadence).mockResolvedValue(cadence);
  let finish = () => {};
  jest.mocked(setCadence).mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        finish = () => resolve({ ...cadence, current: "Weekly" });
      }),
  );
  const onSaved = jest.fn();
  const { result, unmount } = await renderHook(
    () => useRunLessOftenSheet({ scheduleKey: "routine", onSaved }),
    { wrapper: Wrapper },
  );
  await waitFor(() => expect(result.current.cadence).toEqual(cadence));
  await act(() => result.current.choose("weekly"));
  await act(() => {
    result.current.submit();
    result.current.choose("monthly");
    result.current.submit();
  });
  expect(result.current.selected).toBe("weekly");
  expect(setCadence).toHaveBeenCalledTimes(1);
  expect(setCadence).toHaveBeenCalledWith("routine", "weekly");
  await act(async () => finish());
  expect(onSaved).toHaveBeenCalledWith("Weekly");
  await unmount();
  client.clear();
});
