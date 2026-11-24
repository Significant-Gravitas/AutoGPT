import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react-native";
import type { ReactNode } from "react";
import { AppState } from "react-native";

import { useUpdateGate } from "../useUpdateGate";

jest.mock("expo-constants", () => ({ expoConfig: { version: "1.0.0" } }));

describe("useUpdateGate", () => {
  let client: QueryClient;
  let foreground: () => void;
  const originalFetch = globalThis.fetch;
  const fetchMock = jest.fn();

  beforeEach(() => {
    jest.useFakeTimers();
    client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    fetchMock.mockReset();
    globalThis.fetch = fetchMock;
    jest
      .spyOn(AppState, "addEventListener")
      .mockImplementation((_event, listener) => {
        foreground = () => listener("active");
        return { remove: jest.fn() };
      });
  });

  afterEach(async () => {
    await cleanup();
    client.clear();
    globalThis.fetch = originalFetch;
    jest.restoreAllMocks();
    jest.clearAllTimers();
    jest.useRealTimers();
  });

  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  }

  it("lets the app in when the initial check times out", async () => {
    fetchMock.mockImplementation(
      (_url: string, options: RequestInit) =>
        new Promise((_resolve, reject) => {
          options.signal?.addEventListener("abort", () =>
            reject(new Error("Aborted")),
          );
        }),
    );
    const { result } = await renderHook(useUpdateGate, { wrapper: Wrapper });
    expect(result.current).toBe("checking");
    await act(async () => jest.advanceTimersByTime(4_000));
    await act(async () => jest.advanceTimersByTime(0));
    expect(result.current).toBe("ok");
  });

  it("keeps a confirmed block through expired-session cache clearing and a failed recheck", async () => {
    fetchMock.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ mobile_min_version: "2.0.0" }),
    });
    const { result, rerender } = await renderHook(useUpdateGate, {
      wrapper: Wrapper,
    });
    await waitFor(() => expect(result.current).toBe("required"));

    fetchMock.mockRejectedValue(new Error("Offline"));
    await act(async () => client.clear());
    await rerender({});
    expect(result.current).toBe("required");
    await act(async () => foreground());
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    expect(result.current).toBe("required");
  });

  it("checks only on launch, then stays open through a foreground timeout", async () => {
    fetchMock.mockRejectedValueOnce(new Error("Offline"));
    const { result } = await renderHook(useUpdateGate, { wrapper: Wrapper });
    await waitFor(() => expect(result.current).toBe("ok"));

    fetchMock.mockImplementation(
      (_url: string, options: RequestInit) =>
        new Promise((_resolve, reject) => {
          options.signal?.addEventListener("abort", () =>
            reject(new Error("Aborted")),
          );
        }),
    );
    await act(async () => foreground());
    expect(result.current).toBe("ok");
    await act(async () => jest.advanceTimersByTime(4_000));
    expect(result.current).toBe("ok");
  });

  it("waits for the initial answer and lets a later successful check change it", async () => {
    let answer: (value: unknown) => void = () => {};
    fetchMock.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          answer = resolve;
        }),
    );
    const { result } = await renderHook(useUpdateGate, { wrapper: Wrapper });
    expect(result.current).toBe("checking");
    await act(async () =>
      answer({ ok: true, json: async () => ({ mobile_min_version: "2.0.0" }) }),
    );
    await waitFor(() => expect(result.current).toBe("required"));

    fetchMock.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ mobile_min_version: null }),
    });
    await act(async () => foreground());
    await waitFor(() => expect(result.current).toBe("ok"));
  });
});
