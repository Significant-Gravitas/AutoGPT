import { renderHook, waitFor } from "@testing-library/react";
import { StrictMode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useOAuthConnect } from "../useOAuthConnect";

vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({ invalidateQueries: vi.fn() }),
}));

vi.mock("@/lib/oauth-popup", () => ({
  openOAuthPopup: vi.fn(),
  preOpenOAuthPopup: vi.fn(() => null),
  OAUTH_ERROR_POPUP_BLOCKED:
    "Popup blocked — the sign-in window opened in a new tab instead. If you don't see it, allow popups for this site and retry.",
}));

vi.mock("@/app/api/__generated__/endpoints/integrations/integrations", () => ({
  getV1InitiateOauthFlow: vi.fn(),
  postV1ExchangeOauthCodeForTokens: vi.fn(),
  getGetV1ListCredentialsQueryKey: vi.fn(() => ["credentials"]),
}));

const toastMock = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (args: unknown) => toastMock(args),
}));

type ApiErrorShape = Error & { status: number; response: unknown };

function makeApiError(status: number, response: unknown): ApiErrorShape {
  // Reproduces customMutator: `new ApiError(errorMessage, status, responseData)`
  // where errorMessage = responseData.detail. When detail is a non-string
  // (FastAPI 422 array / dict), Error coerces it to "[object Object]".
  const detail = (response as { detail?: unknown })?.detail;
  const message = typeof detail === "string" ? detail : String(detail);
  const err = new Error(message) as ApiErrorShape;
  err.name = "ApiError";
  err.status = status;
  err.response = response;
  return err;
}

async function setupSuccessfulPopup() {
  const { openOAuthPopup } = await import("@/lib/oauth-popup");
  vi.mocked(openOAuthPopup).mockReturnValue({
    promise: Promise.resolve({ code: "auth-code", state: "state-token" }),
    cleanup: { abort: vi.fn() },
    popupBlocked: false,
    fallbackBlocked: false,
  } as unknown as ReturnType<typeof openOAuthPopup>);
}

async function mockInitiateOk() {
  const { getV1InitiateOauthFlow } = await import(
    "@/app/api/__generated__/endpoints/integrations/integrations"
  );
  vi.mocked(getV1InitiateOauthFlow).mockResolvedValue({
    status: 200,
    data: {
      login_url: "https://github.com/login/oauth",
      state_token: "state-token",
    },
  } as never);
}

describe("useOAuthConnect — error toast", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("shows the FastAPI 422 validation message, not [object Object]", async () => {
    await setupSuccessfulPopup();
    await mockInitiateOk();

    const { postV1ExchangeOauthCodeForTokens } = await import(
      "@/app/api/__generated__/endpoints/integrations/integrations"
    );
    vi.mocked(postV1ExchangeOauthCodeForTokens).mockRejectedValue(
      makeApiError(422, {
        detail: [
          {
            type: "missing",
            loc: ["body", "state_token"],
            msg: "Field required",
            input: null,
          },
        ],
      }),
    );

    const { result } = renderHook(() =>
      useOAuthConnect({ provider: "github", onSuccess: vi.fn() }),
    );

    await result.current.connect();

    await waitFor(() => expect(toastMock).toHaveBeenCalled());

    const arg = toastMock.mock.calls[0][0] as {
      title: string;
      description: string;
    };
    expect(arg.title).toBe("OAuth connection failed");
    expect(arg.description).not.toBe("[object Object]");
    expect(arg.description).toContain("Field required");
  });

  it("shows a string detail message unchanged", async () => {
    await setupSuccessfulPopup();
    await mockInitiateOk();

    const { postV1ExchangeOauthCodeForTokens } = await import(
      "@/app/api/__generated__/endpoints/integrations/integrations"
    );
    vi.mocked(postV1ExchangeOauthCodeForTokens).mockRejectedValue(
      makeApiError(400, {
        detail: "OAuth2 callback failed to exchange code for tokens",
      }),
    );

    const { result } = renderHook(() =>
      useOAuthConnect({ provider: "github", onSuccess: vi.fn() }),
    );

    await result.current.connect();

    await waitFor(() => expect(toastMock).toHaveBeenCalled());

    const arg = toastMock.mock.calls[0][0] as { description: string };
    expect(arg.description).toBe(
      "OAuth2 callback failed to exchange code for tokens",
    );
  });

  it("shows the 501 dict detail message for an unconfigured provider", async () => {
    const { getV1InitiateOauthFlow } = await import(
      "@/app/api/__generated__/endpoints/integrations/integrations"
    );
    vi.mocked(getV1InitiateOauthFlow).mockRejectedValue(
      makeApiError(501, {
        detail: {
          message: "Integration with provider 'github' is not configured.",
          hint: "Set client ID and secret in the application's deployment environment",
        },
      }),
    );

    const { result } = renderHook(() =>
      useOAuthConnect({ provider: "github", onSuccess: vi.fn() }),
    );

    await result.current.connect();

    await waitFor(() => expect(toastMock).toHaveBeenCalled());

    const arg = toastMock.mock.calls[0][0] as { description: string };
    expect(arg.description).not.toBe("[object Object]");
    expect(arg.description).toContain("is not configured");
  });

  it("still fires the toast after a StrictMode mount→cleanup→remount", async () => {
    const { getV1InitiateOauthFlow } = await import(
      "@/app/api/__generated__/endpoints/integrations/integrations"
    );
    vi.mocked(getV1InitiateOauthFlow).mockRejectedValue(
      makeApiError(501, { detail: { message: "not configured" } }),
    );

    // StrictMode runs the mount effect, its cleanup, then the effect again on
    // the same instance. The cleanup sets isUnmountedRef to true, so without a
    // reset on (re)mount the catch guard silently swallows the toast.
    const { result } = renderHook(
      () => useOAuthConnect({ provider: "github", onSuccess: vi.fn() }),
      { wrapper: StrictMode },
    );

    await result.current.connect();

    await waitFor(() => expect(toastMock).toHaveBeenCalled());
  });
});

describe("useOAuthConnect — popup window lifecycle", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("pre-opens the window before the initiate await and passes it to openOAuthPopup", async () => {
    await setupSuccessfulPopup();

    const fakeWindow = { closed: false, close: vi.fn() };
    const callOrder: string[] = [];
    const { openOAuthPopup, preOpenOAuthPopup } = await import(
      "@/lib/oauth-popup"
    );
    vi.mocked(preOpenOAuthPopup).mockImplementation(() => {
      callOrder.push("preOpen");
      return fakeWindow as unknown as Window;
    });
    const { getV1InitiateOauthFlow } = await import(
      "@/app/api/__generated__/endpoints/integrations/integrations"
    );
    vi.mocked(getV1InitiateOauthFlow).mockImplementation(async () => {
      callOrder.push("initiate");
      return {
        status: 200,
        data: {
          login_url: "https://github.com/login/oauth",
          state_token: "state-token",
        },
      } as never;
    });

    const { result } = renderHook(() =>
      useOAuthConnect({ provider: "github", onSuccess: vi.fn() }),
    );

    await result.current.connect();

    // The window must be opened synchronously, before the initiate request —
    // after an await iOS Safari blocks every window.open().
    expect(callOrder).toEqual(["preOpen", "initiate"]);
    expect(vi.mocked(openOAuthPopup)).toHaveBeenCalledWith(
      "https://github.com/login/oauth",
      expect.objectContaining({
        stateToken: "state-token",
        preOpenedWindow: fakeWindow,
        useCrossOriginListeners: true,
      }),
    );
  });

  it("closes the pre-opened window when the initiate request fails", async () => {
    const fakeWindow = { closed: false, close: vi.fn() };
    const { preOpenOAuthPopup } = await import("@/lib/oauth-popup");
    vi.mocked(preOpenOAuthPopup).mockReturnValue(
      fakeWindow as unknown as Window,
    );

    const { getV1InitiateOauthFlow } = await import(
      "@/app/api/__generated__/endpoints/integrations/integrations"
    );
    vi.mocked(getV1InitiateOauthFlow).mockRejectedValue(
      makeApiError(501, { detail: { message: "not configured" } }),
    );

    const { result } = renderHook(() =>
      useOAuthConnect({ provider: "github", onSuccess: vi.fn() }),
    );

    await result.current.connect();

    await waitFor(() => expect(toastMock).toHaveBeenCalled());
    expect(fakeWindow.close).toHaveBeenCalled();
  });

  it("closes the pre-opened window and skips adoption when unmounted mid-initiation", async () => {
    const fakeWindow = { closed: false, close: vi.fn() };
    const { openOAuthPopup, preOpenOAuthPopup } = await import(
      "@/lib/oauth-popup"
    );
    vi.mocked(preOpenOAuthPopup).mockReturnValue(
      fakeWindow as unknown as Window,
    );

    let resolveInitiate: (value: unknown) => void = () => {};
    const { getV1InitiateOauthFlow } = await import(
      "@/app/api/__generated__/endpoints/integrations/integrations"
    );
    vi.mocked(getV1InitiateOauthFlow).mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveInitiate = resolve;
        }) as never,
    );

    const { result, unmount } = renderHook(() =>
      useOAuthConnect({ provider: "github", onSuccess: vi.fn() }),
    );

    void result.current.connect();

    // Unmount while the initiate request is still in flight — the cleanup
    // must close the pre-opened window immediately.
    unmount();
    expect(fakeWindow.close).toHaveBeenCalled();

    // When the request resolves, the stale continuation must not adopt the
    // window or surface anything.
    resolveInitiate({
      status: 200,
      data: {
        login_url: "https://github.com/login/oauth",
        state_token: "state-token",
      },
    });
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(vi.mocked(openOAuthPopup)).not.toHaveBeenCalled();
    expect(toastMock).not.toHaveBeenCalled();
  });

  it("warns the user when the browser blocked the popup", async () => {
    await mockInitiateOk();

    const { openOAuthPopup, preOpenOAuthPopup } = await import(
      "@/lib/oauth-popup"
    );
    vi.mocked(preOpenOAuthPopup).mockReturnValue(null);
    vi.mocked(openOAuthPopup).mockReturnValue({
      promise: new Promise(() => {}), // flow stays in flight
      cleanup: { abort: vi.fn() },
      popupBlocked: true,
      fallbackBlocked: false,
    } as unknown as ReturnType<typeof openOAuthPopup>);

    const { result } = renderHook(() =>
      useOAuthConnect({ provider: "github", onSuccess: vi.fn() }),
    );

    // The mocked popup promise never settles, so connect() stays in flight —
    // don't await it; just wait for the warning toast it fires synchronously
    // after openOAuthPopup returns.
    void result.current.connect();

    await waitFor(() => expect(toastMock).toHaveBeenCalled());
    const arg = toastMock.mock.calls[0][0] as { title: string };
    expect(arg.title).toBe("Popup blocked");
  });

  it("shows only the failure toast when both popup and fallback tab are blocked", async () => {
    await mockInitiateOk();

    const { openOAuthPopup, preOpenOAuthPopup } = await import(
      "@/lib/oauth-popup"
    );
    vi.mocked(preOpenOAuthPopup).mockReturnValue(null);
    vi.mocked(openOAuthPopup).mockImplementation(
      () =>
        ({
          promise: Promise.reject(
            new Error(
              "Popup blocked — allow popups for this site and try again.",
            ),
          ),
          cleanup: { abort: vi.fn() },
          popupBlocked: true,
          fallbackBlocked: true,
        }) as unknown as ReturnType<typeof openOAuthPopup>,
    );

    const { result } = renderHook(() =>
      useOAuthConnect({ provider: "github", onSuccess: vi.fn() }),
    );

    await result.current.connect();

    await waitFor(() => expect(toastMock).toHaveBeenCalled());
    // Exactly one toast — the "opened in a new tab" hint would contradict it.
    expect(toastMock).toHaveBeenCalledTimes(1);
    const arg = toastMock.mock.calls[0][0] as {
      title: string;
      description: string;
    };
    expect(arg.title).toBe("OAuth connection failed");
    expect(arg.description).toBe(
      "Popup blocked — allow popups for this site and try again.",
    );
  });

  it("ignores a re-entrant connect() while a flow is in flight", async () => {
    const { preOpenOAuthPopup } = await import("@/lib/oauth-popup");
    vi.mocked(preOpenOAuthPopup).mockReturnValue(null);

    let resolveInitiate: (value: unknown) => void = () => {};
    const { getV1InitiateOauthFlow } = await import(
      "@/app/api/__generated__/endpoints/integrations/integrations"
    );
    vi.mocked(getV1InitiateOauthFlow).mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveInitiate = resolve;
        }) as never,
    );

    const { result } = renderHook(() =>
      useOAuthConnect({ provider: "github", onSuccess: vi.fn() }),
    );

    // A rapid double-click fires the second call while the first is still
    // awaiting the initiate request — it must return without side effects.
    const first = result.current.connect();
    await result.current.connect();

    expect(vi.mocked(preOpenOAuthPopup)).toHaveBeenCalledTimes(1);
    expect(vi.mocked(getV1InitiateOauthFlow)).toHaveBeenCalledTimes(1);

    // Let the first flow finish so the test doesn't leak a pending flow.
    await setupSuccessfulPopup();
    resolveInitiate({
      status: 200,
      data: {
        login_url: "https://github.com/login/oauth",
        state_token: "state-token",
      },
    });
    await first;
  });
});
