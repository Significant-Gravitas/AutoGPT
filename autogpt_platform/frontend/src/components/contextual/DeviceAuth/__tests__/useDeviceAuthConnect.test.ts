import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { createElement, type ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  postV1InitiateDeviceCodeOauthFlow,
  postV1PollDeviceCodeOauthFlowForCompletion,
} from "@/app/api/__generated__/endpoints/integrations/integrations";

import { useDeviceAuthConnect } from "../useDeviceAuthConnect";

vi.mock("@/app/api/__generated__/endpoints/integrations/integrations", () => ({
  postV1InitiateDeviceCodeOauthFlow: vi.fn(),
  postV1PollDeviceCodeOauthFlowForCompletion: vi.fn(),
  getGetV1ListCredentialsQueryKey: () => ["credentials"],
}));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: vi.fn(),
}));

const initiate = vi.mocked(postV1InitiateDeviceCodeOauthFlow);
const poll = vi.mocked(postV1PollDeviceCodeOauthFlowForCompletion);

function initiated(overrides: Record<string, unknown> = {}) {
  return {
    status: 200,
    data: {
      state_token: "state_1",
      user_code: "glow-relish-chaste-soft",
      verification_url: "https://app.link.com/device/setup",
      verification_url_complete: "https://app.link.com/device/setup?code=glow",
      expires_in: 600,
      interval: 5,
      ...overrides,
    },
  } as never;
}

function makeWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return function Wrapper({ children }: { children: ReactNode }) {
    return createElement(QueryClientProvider, { client }, children);
  };
}

function render(onSuccess = vi.fn()) {
  return renderHook(
    () => useDeviceAuthConnect({ provider: "stripe_link", onSuccess }),
    { wrapper: makeWrapper() },
  );
}

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  vi.clearAllMocks();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("useDeviceAuthConnect", () => {
  it("surfaces the code phrase and the completed verification URL", async () => {
    initiate.mockResolvedValue(initiated());
    poll.mockResolvedValue({
      status: 200,
      data: { status: "pending" },
    } as never);

    const { result } = render();
    await act(async () => {
      await result.current.connect();
    });

    expect(result.current.userCode).toBe("glow-relish-chaste-soft");
    // The completed URL carries the code, so prefer it when present.
    expect(result.current.verificationUrl).toBe(
      "https://app.link.com/device/setup?code=glow",
    );
    expect(result.current.phase).toBe("polling");
  });

  it("falls back to the plain verification URL when there is no complete form", async () => {
    initiate.mockResolvedValue(initiated({ verification_url_complete: null }));
    poll.mockResolvedValue({
      status: 200,
      data: { status: "pending" },
    } as never);

    const { result } = render();
    await act(async () => {
      await result.current.connect();
    });

    expect(result.current.verificationUrl).toBe(
      "https://app.link.com/device/setup",
    );
  });

  it("reports an error phase when initiation does not return 200", async () => {
    // The generated client returns a union including error responses; the hook
    // must narrow on the discriminant rather than assume success.
    initiate.mockResolvedValue({
      status: 401,
      data: { detail: "nope" },
    } as never);

    const { result } = render();
    await act(async () => {
      await result.current.connect();
    });

    expect(result.current.phase).toBe("error");
    expect(poll).not.toHaveBeenCalled();
  });

  it("does not strand the previous poll loop when connect is called twice", async () => {
    initiate.mockResolvedValue(initiated());
    poll.mockResolvedValue({
      status: 200,
      data: { status: "pending" },
    } as never);

    const { result } = render();
    await act(async () => {
      await result.current.connect();
    });
    await act(async () => {
      await result.current.connect();
    });

    // Two initiations, but the first loop was cancelled — so advancing time
    // must not produce polls from both.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000);
    });
    expect(initiate).toHaveBeenCalledTimes(2);
    expect(poll.mock.calls.length).toBeLessThanOrEqual(1);
  });

  it("clamps an out-of-range interval from the provider", async () => {
    // The interval drives a setTimeout: 0 would spin the loop.
    initiate.mockResolvedValue(initiated({ interval: 0 }));
    poll.mockResolvedValue({
      status: 200,
      data: { status: "pending" },
    } as never);

    const { result } = render();
    await act(async () => {
      await result.current.connect();
    });

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_000);
    });
    // Unclamped, a 0 interval spins: this would be hundreds of calls in 2s.
    expect(poll.mock.calls.length).toBeLessThanOrEqual(1);
  });

  it("completes and notifies once the provider approves", async () => {
    initiate.mockResolvedValue(initiated({ interval: 1 }));
    poll.mockResolvedValue({
      status: 200,
      data: { status: "approved", credentials: { id: "cred_1" } },
    } as never);
    const onSuccess = vi.fn();

    const { result } = render(onSuccess);
    await act(async () => {
      await result.current.connect();
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_500);
    });

    await waitFor(() => expect(result.current.phase).toBe("done"));
    expect(onSuccess).toHaveBeenCalled();
  });

  it("stops polling when the provider denies", async () => {
    initiate.mockResolvedValue(initiated({ interval: 1 }));
    poll.mockResolvedValue({
      status: 200,
      data: { status: "denied" },
    } as never);

    const { result } = render();
    await act(async () => {
      await result.current.connect();
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_500);
    });

    await waitFor(() => expect(result.current.phase).toBe("error"));
    const callsAfterTerminal = poll.mock.calls.length;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000);
    });
    expect(poll.mock.calls.length).toBe(callsAfterTerminal);
  });

  it("works again after a remount", async () => {
    // The unmount ref survives a remount, so a stale `true` would silently
    // no-op every callback and the flow would appear dead the second time.
    initiate.mockResolvedValue(initiated());
    poll.mockResolvedValue({
      status: 200,
      data: { status: "pending" },
    } as never);

    const first = render();
    await act(async () => {
      await first.result.current.connect();
    });
    first.unmount();

    const second = render();
    await act(async () => {
      await second.result.current.connect();
    });

    expect(second.result.current.userCode).toBe("glow-relish-chaste-soft");
    expect(second.result.current.phase).toBe("polling");
  });
});
