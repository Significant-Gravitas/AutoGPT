import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react-native";
import type { ReactNode } from "react";

import { getExpertDetail, type ExpertDetail } from "@/lib/api/expertActivity";
import { updateExpertBudget, updateExpertMode } from "@/lib/api/expertManage";

import { useManageExpert } from "../useManageExpert";

jest.mock("expo-router", () => ({ router: {} }));
jest.mock("@/lib/session", () => ({ sessionGeneration: () => 1 }));
jest.mock("@/lib/api/flags", () => ({
  flagKeys: { all: ["flags"] },
  getFlags: async () => ({}),
  isFlagOn: () => true,
}));
jest.mock("@/lib/api/expertActivity", () => ({
  ...jest.requireActual("@/lib/api/expertActivity"),
  getExpertDetail: jest.fn(),
}));
jest.mock("@/lib/api/expertManage", () => ({
  updateExpertBudget: jest.fn(),
  updateExpertMode: jest.fn(),
}));

const expert: ExpertDetail = {
  id: "12345678-1234-4234-8234-123456789abc",
  name: "Mia",
  identity: "Bookkeeper",
  voice_preferences: "",
  boundaries: "",
  weekly_budget: 2500,
  weekly_spend: 0,
  autopilot_mode: null,
  source_template_id: null,
  schedules_paused_at: null,
  workflows: [],
};

let client: QueryClient;

function Wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

function useExpert() {
  return useManageExpert({ routeExpertID: expert.id, routeFrom: null });
}

beforeEach(() => {
  client = new QueryClient({
    defaultOptions: {
      queries: { gcTime: Infinity },
      mutations: { gcTime: Infinity },
    },
  });
  jest.mocked(getExpertDetail).mockResolvedValue(expert);
});

afterEach(() => {
  client.clear();
  jest.clearAllMocks();
});

test("one approval save runs at a time and a refused save rolls back", async () => {
  let fail = () => {};
  jest.mocked(updateExpertMode).mockImplementationOnce(
    () =>
      new Promise((_resolve, reject) => {
        fail = () => reject(new Error("Save refused"));
      }),
  );
  const screen = await renderHook(useExpert, { wrapper: Wrapper });
  await waitFor(() => expect(screen.result.current.expert).toEqual(expert));
  await act(() => {
    screen.result.current.chooseMode("ask_first");
    screen.result.current.chooseMode("auto");
  });
  expect(updateExpertMode).toHaveBeenCalledTimes(1);
  expect(updateExpertMode).toHaveBeenCalledWith(expert.id, "ask_first");
  await waitFor(() =>
    expect(screen.result.current.expert?.autopilot_mode).toBe("ask_first"),
  );
  await act(async () => fail());
  await waitFor(() =>
    expect(screen.result.current.expert?.autopilot_mode).toBeNull(),
  );
});

test("a budget save leaves a later sheet open and uses the returned effective cap", async () => {
  let finish = () => {};
  jest.mocked(updateExpertBudget).mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        finish = () => resolve({ ...expert, weekly_budget: null });
      }),
  );
  const screen = await renderHook(useExpert, { wrapper: Wrapper });
  await waitFor(() => expect(screen.result.current.expert).toEqual(expert));
  await act(() => screen.result.current.openBudget());
  await act(() => {
    void screen.result.current.saveBudget(0);
  });
  expect(updateExpertBudget).toHaveBeenCalledWith(expert.id, 0);
  expect(screen.result.current.expert?.weekly_budget).toBe(2500);
  await act(() => {
    screen.result.current.closeSheet();
    screen.result.current.openProfile();
  });
  await act(async () => finish());
  await waitFor(() =>
    expect(screen.result.current.expert?.weekly_budget).toBeNull(),
  );
  expect(screen.result.current.sheet).toBe("profile");
});
