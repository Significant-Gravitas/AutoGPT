import { renderHook, waitFor } from "@testing-library/react";
import { StrictMode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useOAuthConnect } from "../useOAuthConnect";

vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({ invalidateQueries: vi.fn() }),
}));

vi.mock("@/lib/oauth-popup", () => ({
  openOAuthPopup: vi.fn(),
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
