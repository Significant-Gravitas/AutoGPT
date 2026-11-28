import { QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react-native";
import * as WebBrowser from "expo-web-browser";
import type { ReactNode } from "react";

import { ApiError } from "@/lib/api/client";
import {
  cancelCodexLogin,
  getAyrshareLink,
  getCodexLoginStatus,
  startProviderLogin,
} from "@/lib/api/integrations";
import { queryClient, sessionGeneration } from "@/lib/session";

import { useDeviceCodeSheet } from "../components/DeviceCodeSheet/useDeviceCodeSheet";
import { useConnect } from "../useConnect";

jest.mock("@/lib/session", () => {
  const { QueryClient } = jest.requireActual("@tanstack/react-query");
  return {
    queryClient: new QueryClient({
      defaultOptions: {
        queries: { gcTime: Infinity },
        mutations: { gcTime: Infinity },
      },
    }),
    sessionGeneration: jest.fn(),
  };
});
jest.mock("@/lib/api/integrations", () => ({
  ...jest.requireActual("@/lib/api/integrations"),
  cancelCodexLogin: jest.fn(),
  getAyrshareLink: jest.fn(),
  getCodexLoginStatus: jest.fn(),
  startProviderLogin: jest.fn(),
}));
jest.mock("expo-web-browser", () => ({
  openBrowserAsync: jest.fn(),
  WebBrowserResultType: { OPENED: "opened" },
}));

function Wrapper({ children }: { children: ReactNode }) {
  return (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

function login(id: string) {
  return {
    loginURL: `https://app.example/api/proxy/api/integrations/codex/device-login/${id}#verification_url=https%3A%2F%2Fauth.openai.com%2Fcodex%2Fdevice&user_code=${id}`,
    state: "state",
    cancelURL: null,
  };
}

function useDeviceLogin(credentialID?: string) {
  return useDeviceCodeSheet({
    credentialID,
    onSaved: jest.fn(),
    onClose: jest.fn(),
    onPlanRequired: jest.fn(),
  });
}

function useAyrshare(onConnected: jest.Mock) {
  return useConnect({
    origin: { sessionID: null, cardKey: null, expertID: null },
    onConnected,
  });
}

beforeEach(() => {
  jest.mocked(sessionGeneration).mockReturnValue(1);
  jest
    .mocked(getCodexLoginStatus)
    .mockResolvedValue({ status: "pending", error: null });
  jest.mocked(cancelCodexLogin).mockResolvedValue(undefined);
});

afterEach(() => {
  queryClient.clear();
  jest.resetAllMocks();
});

test("a deleted reconnect target falls back to a fresh device login", async () => {
  jest
    .mocked(startProviderLogin)
    .mockRejectedValueOnce(new ApiError(404, "Credential to upgrade not found"))
    .mockResolvedValueOnce(login("fresh"));
  const { result } = await renderHook(() => useDeviceLogin("deleted"), {
    wrapper: Wrapper,
  });
  await waitFor(() => expect(result.current.userCode).toBe("fresh"));
  expect(startProviderLogin).toHaveBeenNthCalledWith(1, "codex", {
    scopes: [],
    credentialID: "deleted",
  });
  expect(startProviderLogin).toHaveBeenNthCalledWith(2, "codex", {
    scopes: [],
  });
  await act(() => result.current.close());
});

test("a late start cancels only itself and keeps the newer login tracked", async () => {
  let finishFirst = () => {};
  jest
    .mocked(startProviderLogin)
    .mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finishFirst = () => resolve(login("first"));
        }),
    )
    .mockResolvedValueOnce(login("second"));

  const first = await renderHook(() => useDeviceLogin(), { wrapper: Wrapper });
  expect(startProviderLogin).toHaveBeenCalledTimes(1);
  await act(() => first.result.current.close());
  await first.unmount();
  const second = await renderHook(() => useDeviceLogin(), { wrapper: Wrapper });
  await waitFor(() => expect(second.result.current.userCode).toBe("second"));
  await act(async () => finishFirst());
  expect(cancelCodexLogin).toHaveBeenLastCalledWith("first");
  await act(() => second.result.current.close());
  expect(cancelCodexLogin).toHaveBeenLastCalledWith("second");
});

test("opening Ayrshare does not mark a social account as linked", async () => {
  const onConnected = jest.fn();
  const sso_url = "https://ayrshare.example/link";
  jest.mocked(getAyrshareLink).mockResolvedValue({ sso_url });
  jest.mocked(WebBrowser.openBrowserAsync).mockResolvedValue({
    type: WebBrowser.WebBrowserResultType.OPENED,
  });
  const { result } = await renderHook(() => useAyrshare(onConnected));
  await act(() => result.current.linkAyrshare("social", "Ayrshare"));
  expect(WebBrowser.openBrowserAsync).toHaveBeenCalledWith(sso_url);
  expect(onConnected).not.toHaveBeenCalled();
});

test("an Ayrshare link from a previous session cannot open a browser", async () => {
  let finish = () => {};
  jest.mocked(getAyrshareLink).mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        finish = () => resolve({ sso_url: "https://ayrshare.example/link" });
      }),
  );
  const onConnected = jest.fn();
  const { result } = await renderHook(() => useAyrshare(onConnected));
  await act(() => {
    void result.current.linkAyrshare("social", "Ayrshare");
  });
  jest.mocked(sessionGeneration).mockReturnValue(2);
  await act(async () => finish());
  expect(WebBrowser.openBrowserAsync).not.toHaveBeenCalled();
  expect(onConnected).not.toHaveBeenCalled();
});
