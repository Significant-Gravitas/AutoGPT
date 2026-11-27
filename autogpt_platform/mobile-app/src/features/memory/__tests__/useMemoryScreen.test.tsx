import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react-native";
import type { ReactNode } from "react";
import { Alert } from "react-native";

import {
  eraseMemory,
  getMemoryOverview,
  getUnderstanding,
  listMemoryFacts,
  type MemoryUnderstanding,
} from "@/lib/api/memory";

import { useMemoryScreen } from "../useMemoryScreen";

jest.mock("expo-router", () => ({ router: {} }));
jest.mock("@/lib/session", () => ({ sessionGeneration: () => 1 }));
jest.mock("@/lib/haptics", () => ({ tapMenu: jest.fn() }));
jest.mock("@/lib/api/memory", () => ({
  ...jest.requireActual("@/lib/api/memory"),
  eraseMemory: jest.fn(),
  getMemoryOverview: jest.fn(),
  getUnderstanding: jest.fn(),
  listMemoryFacts: jest.fn(),
}));

const overview = { facts: 1, entities: 2, episodes: 3 };
const fact = {
  uuid: "a",
  fact: "Runs payroll",
  name: null,
  source: "",
  target: "",
  created_at: null,
};
const understanding = { business_name: "Acme" } as MemoryUnderstanding;
let client: QueryClient;

function Wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

beforeEach(() => {
  client = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: Infinity } },
  });
  jest.mocked(getMemoryOverview).mockResolvedValue(overview);
  jest.mocked(getUnderstanding).mockResolvedValue(understanding);
  jest.mocked(listMemoryFacts).mockResolvedValue({ items: [fact] });
});

afterEach(() => {
  client.clear();
  jest.resetAllMocks();
  jest.restoreAllMocks();
});

test("an overview failure can be retried while facts stay visible", async () => {
  jest.mocked(getMemoryOverview).mockRejectedValueOnce(new Error("Offline"));
  const { result } = await renderHook(useMemoryScreen, { wrapper: Wrapper });
  await waitFor(() => expect(result.current.isOverviewError).toBe(true));
  expect(result.current.facts).toEqual([fact]);
  expect(result.current.overview).toBeNull();
  await act(() => result.current.retry());
  await waitFor(() => expect(result.current.overview).toEqual(overview));
});

test("a successful erase clears visible data even if the refresh fails", async () => {
  jest.mocked(eraseMemory).mockResolvedValue({ erased: true });
  jest.spyOn(Alert, "alert").mockImplementation((_title, _message, buttons) => {
    buttons?.find((button) => button.style === "destructive")?.onPress?.();
  });
  const { result } = await renderHook(useMemoryScreen, { wrapper: Wrapper });
  await waitFor(() => expect(result.current.overview).toEqual(overview));
  expect(result.current.understandingRows).toEqual([
    { label: "Business", value: "Acme" },
  ]);
  expect(result.current.facts).toEqual([fact]);

  const offline = new Error("Offline");
  jest.mocked(getMemoryOverview).mockRejectedValue(offline);
  jest.mocked(getUnderstanding).mockRejectedValue(offline);
  jest.mocked(listMemoryFacts).mockRejectedValue(offline);
  await act(() => result.current.confirmErase());
  await waitFor(() => expect(result.current.isRetrying).toBe(false));
  expect(result.current.facts).toEqual([]);
  expect(result.current.understandingRows).toEqual([]);
  expect(result.current.overview).toEqual({
    facts: 0,
    entities: 0,
    episodes: 0,
  });
});
