import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  getV1InitiateOauthFlow,
  postV1ExchangeOauthCodeForTokens,
} from "@/app/api/__generated__/endpoints/integrations/integrations";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useOAuthConnect } from "../useOAuthConnect";

vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({ invalidateQueries: vi.fn() }),
}));
vi.mock("@/app/api/__generated__/endpoints/integrations/integrations", () => ({
  getV1InitiateOauthFlow: vi.fn(),
  postV1ExchangeOauthCodeForTokens: vi.fn(),
}));
vi.mock("@/lib/react-query/invalidateConnections", () => ({
  invalidateConnectionQueries: vi.fn(),
}));
vi.mock("@/components/molecules/Toast/use-toast", () => ({ toast: vi.fn() }));
vi.mock("@/components/contextual/PaywallToast/helpers", () => ({
  toastIfPlanRequired: () => false,
}));
vi.mock("@/providers/agent-credentials/credentials-provider", async () => {
  const { createContext } = await import("react");
  return { CredentialsActionsContext: createContext(null) };
});

function popupWindow() {
  return { closed: false, location: { href: "" }, close: vi.fn() };
}

describe("useOAuthConnect with the real popup lifecycle", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    localStorage.clear();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(null, { status: 204 })),
    );
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.restoreAllMocks();
    vi.clearAllMocks();
    vi.unstubAllGlobals();
  });

  it("releases ChatGPT pending state on close and immediately allows a fresh attempt", async () => {
    const firstPopup = popupWindow();
    const secondPopup = popupWindow();
    vi.spyOn(window, "open")
      .mockReturnValueOnce(firstPopup as unknown as Window)
      .mockReturnValueOnce(secondPopup as unknown as Window);
    vi.mocked(getV1InitiateOauthFlow)
      .mockResolvedValueOnce({
        status: 200,
        data: {
          login_url: `${window.location.origin}/api/proxy/api/integrations/codex/device-login/first`,
          state_token: "first-state",
          cancel_url:
            "/api/proxy/api/integrations/codex/device-login/first/cancel",
        },
      } as never)
      .mockResolvedValueOnce({
        status: 200,
        data: {
          login_url: `${window.location.origin}/api/proxy/api/integrations/codex/device-login/second`,
          state_token: "second-state",
          cancel_url:
            "/api/proxy/api/integrations/codex/device-login/second/cancel",
        },
      } as never);
    const onSuccess = vi.fn();
    const { result, unmount } = renderHook(() =>
      useOAuthConnect({ provider: "codex", onSuccess }),
    );
    let first: Promise<void>;
    await act(async () => {
      first = result.current.connect();
    });
    expect(result.current.isPending).toBe(true);
    firstPopup.closed = true;

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000);
    });

    expect(result.current.isPending).toBe(false);
    await first!;
    expect(onSuccess).not.toHaveBeenCalled();
    expect(postV1ExchangeOauthCodeForTokens).not.toHaveBeenCalled();
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(toast).toHaveBeenCalledWith(
      expect.objectContaining({ description: "Sign-in window was closed" }),
    );
    let second: Promise<void>;
    await act(async () => {
      second = result.current.connect();
    });
    expect(getV1InitiateOauthFlow).toHaveBeenCalledTimes(2);
    expect(result.current.isPending).toBe(true);
    expect(secondPopup.location.href).toContain("/device-login/second");

    unmount();
    await second!;
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("still waits for a third-party OAuth result when COOP reports its window closed", async () => {
    const popup = popupWindow();
    vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window);
    vi.mocked(getV1InitiateOauthFlow).mockResolvedValue({
      status: 200,
      data: {
        login_url: `${window.location.origin}/api/oauth/redirect`,
        state_token: "third-party",
      },
    } as never);
    const { result, unmount } = renderHook(() =>
      useOAuthConnect({ provider: "github", onSuccess: vi.fn() }),
    );
    let pending: Promise<void>;
    await act(async () => {
      pending = result.current.connect();
    });
    popup.closed = true;

    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000);
    });

    expect(result.current.isPending).toBe(true);
    expect(postV1ExchangeOauthCodeForTokens).not.toHaveBeenCalled();
    expect(toast).not.toHaveBeenCalled();
    unmount();
    await pending!;
    expect(vi.getTimerCount()).toBe(0);
  });

  it("exchanges a completed ChatGPT callback once even when the popup has closed", async () => {
    const popup = popupWindow();
    vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window);
    vi.mocked(getV1InitiateOauthFlow).mockResolvedValue({
      status: 200,
      data: {
        login_url: `${window.location.origin}/api/proxy/api/integrations/codex/device-login/completed`,
        state_token: "completed-state",
        cancel_url:
          "/api/proxy/api/integrations/codex/device-login/completed/cancel",
      },
    } as never);
    vi.mocked(postV1ExchangeOauthCodeForTokens).mockResolvedValue({
      status: 200,
      data: { id: "connected-credential" },
    } as never);
    const onSuccess = vi.fn();
    const { result } = renderHook(() =>
      useOAuthConnect({ provider: "codex", onSuccess }),
    );
    let pending: Promise<void>;
    await act(async () => {
      pending = result.current.connect();
    });
    localStorage.setItem(
      "oauth_popup_result_completed-state",
      JSON.stringify({
        message_type: "oauth_popup_result",
        success: true,
        code: "completed",
        state: "completed-state",
      }),
    );
    popup.closed = true;

    await act(async () => {
      await vi.advanceTimersByTimeAsync(500);
      await pending;
    });

    expect(result.current.isPending).toBe(false);
    expect(postV1ExchangeOauthCodeForTokens).toHaveBeenCalledExactlyOnceWith(
      "codex",
      {
        code: "completed",
        state_token: "completed-state",
      },
    );
    expect(onSuccess).toHaveBeenCalledExactlyOnceWith({
      id: "connected-credential",
    });
    expect(fetch).not.toHaveBeenCalled();
    expect(vi.getTimerCount()).toBe(0);
  });
});
