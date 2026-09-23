import { cleanup, render } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mockToast = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return {
    ...actual,
    toast: (...args: unknown[]) => mockToast(...args),
  };
});

const mockUseGetV2GetCopilotUsage = vi.fn();
vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  useGetV2GetCopilotUsage: (...args: unknown[]) =>
    mockUseGetV2GetCopilotUsage(...args),
}));

const mockUseGetSubscriptionStatus = vi.fn();
vi.mock("@/app/api/__generated__/endpoints/credits/credits", () => ({
  useGetSubscriptionStatus: (...args: unknown[]) =>
    mockUseGetSubscriptionStatus(...args),
}));

// The switch-connection mechanics are borrowed from the provider-limit
// dialog's hook. Stubbed here so these tests stay about the gate: what it
// asks that hook for, and what it hands the dialog back.
const mockUseProviderLimitDialog = vi.fn();
vi.mock("../../ProviderLimitDialog/useProviderLimitDialog", () => ({
  useProviderLimitDialog: (...args: unknown[]) =>
    mockUseProviderLimitDialog(...args),
}));

// Capture props the dialog was rendered with so we can assert on them.
const dialogSpy = vi.fn();
vi.mock("../RateLimitResetDialog", () => ({
  RateLimitResetDialog: (props: {
    isOpen: boolean;
    onClose: () => void;
    resetsAt?: string | Date | null;
    tier?: string | null;
    alternative?: { display_name: string } | null;
    onContinue?: () => void;
    isSwitching?: boolean;
  }) => {
    dialogSpy(props);
    return <div data-testid="reset-dialog" data-open={String(props.isOpen)} />;
  },
}));

import { RateLimitGate } from "../RateLimitGate";

function noAlternative() {
  mockUseProviderLimitDialog.mockReturnValue({
    alternative: null,
    continueHere: vi.fn(),
    isSwitching: false,
    isLoadingOffers: false,
    failedToLoadOffers: false,
    retryOffers: vi.fn(),
    resetHint: null,
  });
}

beforeEach(() => {
  noAlternative();
});

afterEach(() => {
  cleanup();
  mockToast.mockReset();
  mockUseGetV2GetCopilotUsage.mockReset();
  mockUseGetSubscriptionStatus.mockReset();
  mockUseProviderLimitDialog.mockReset();
  dialogSpy.mockReset();
});

function setSubscription(tier: string | null = null) {
  mockUseGetSubscriptionStatus.mockReturnValue({
    data: tier,
    isSuccess: tier !== null,
    isError: false,
  });
}

describe("RateLimitGate", () => {
  it("disables the usage query when no rate-limit message is present", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: undefined,
      isSuccess: false,
      isError: false,
    });
    setSubscription();

    render(<RateLimitGate rateLimitMessage={null} onDismiss={vi.fn()} />);

    expect(mockUseGetV2GetCopilotUsage).toHaveBeenCalled();
    const [config] = mockUseGetV2GetCopilotUsage.mock.calls[0] as [
      { query?: { enabled?: boolean } },
    ];
    expect(config?.query?.enabled).toBe(false);
  });

  it("enables the usage query once a rate-limit message arrives", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: undefined,
      isSuccess: false,
      isError: false,
    });
    setSubscription();

    render(
      <RateLimitGate
        rateLimitMessage="You've hit your usage limit"
        onDismiss={vi.fn()}
      />,
    );

    const [config] = mockUseGetV2GetCopilotUsage.mock.calls[0] as [
      { query?: { enabled?: boolean } },
    ];
    expect(config?.query?.enabled).toBe(true);
  });

  it("disables the subscription query when no rate-limit message is present", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: undefined,
      isSuccess: false,
      isError: false,
    });
    setSubscription();

    render(<RateLimitGate rateLimitMessage={null} onDismiss={vi.fn()} />);

    expect(mockUseGetSubscriptionStatus).toHaveBeenCalled();
    const [config] = mockUseGetSubscriptionStatus.mock.calls[0] as [
      { query?: { enabled?: boolean } },
    ];
    expect(config?.query?.enabled).toBe(false);
  });

  it("opens the reset dialog when usage data is available", () => {
    const future = new Date(Date.now() + 4 * 60 * 60 * 1000).toISOString();
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: { daily: { percent_used: 100, resets_at: future }, weekly: null },
      isSuccess: true,
      isError: false,
    });
    setSubscription("PRO");

    render(
      <RateLimitGate rateLimitMessage="limit reached" onDismiss={vi.fn()} />,
    );

    expect(dialogSpy).toHaveBeenCalled();
    const lastProps = dialogSpy.mock.calls.at(-1)?.[0];
    expect(lastProps.isOpen).toBe(true);
    expect(mockToast).not.toHaveBeenCalled();
  });

  it("forwards the user's subscription tier to the dialog", () => {
    const future = new Date(Date.now() + 4 * 60 * 60 * 1000).toISOString();
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: { daily: { percent_used: 100, resets_at: future }, weekly: null },
      isSuccess: true,
      isError: false,
    });
    setSubscription("MAX");

    render(
      <RateLimitGate rateLimitMessage="limit reached" onDismiss={vi.fn()} />,
    );

    const lastProps = dialogSpy.mock.calls.at(-1)?.[0];
    expect(lastProps.tier).toBe("MAX");
  });

  it("passes null tier when subscription status is unavailable", () => {
    const future = new Date(Date.now() + 4 * 60 * 60 * 1000).toISOString();
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: { daily: { percent_used: 100, resets_at: future }, weekly: null },
      isSuccess: true,
      isError: false,
    });
    setSubscription(null);

    render(
      <RateLimitGate rateLimitMessage="limit reached" onDismiss={vi.fn()} />,
    );

    const lastProps = dialogSpy.mock.calls.at(-1)?.[0];
    expect(lastProps.tier).toBeNull();
  });

  it("falls back to a toast when usage query errors", () => {
    const onDismiss = vi.fn();
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: undefined,
      isSuccess: false,
      isError: true,
    });
    setSubscription();

    render(
      <RateLimitGate
        rateLimitMessage="You've hit your usage limit"
        onDismiss={onDismiss}
      />,
    );

    expect(mockToast).toHaveBeenCalledTimes(1);
    const toastArg = mockToast.mock.calls[0][0] as {
      title: string;
      variant: string;
    };
    expect(toastArg.title).toBe("Usage limit reached");
    expect(toastArg.variant).toBe("destructive");
    expect(onDismiss).toHaveBeenCalledTimes(1);

    const lastProps = dialogSpy.mock.calls.at(-1)?.[0];
    expect(lastProps.isOpen).toBe(false);
  });

  it("does not fire the fallback toast when no rate-limit message is present", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: undefined,
      isSuccess: false,
      isError: true,
    });
    setSubscription();

    render(<RateLimitGate rateLimitMessage={null} onDismiss={vi.fn()} />);

    expect(mockToast).not.toHaveBeenCalled();
  });

  it("keeps the dialog closed while the usage query is still loading", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: undefined,
      isSuccess: false,
      isError: false,
    });
    setSubscription();

    render(
      <RateLimitGate rateLimitMessage="limit reached" onDismiss={vi.fn()} />,
    );

    const lastProps = dialogSpy.mock.calls.at(-1)?.[0];
    expect(lastProps.isOpen).toBe(false);
    expect(mockToast).not.toHaveBeenCalled();
  });

  it("forwards daily resets_at timestamp to the dialog", () => {
    const future = new Date(Date.now() + 4 * 60 * 60 * 1000).toISOString();
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: { daily: { percent_used: 100, resets_at: future }, weekly: null },
      isSuccess: true,
      isError: false,
    });
    setSubscription();

    render(
      <RateLimitGate rateLimitMessage="limit reached" onDismiss={vi.fn()} />,
    );

    const lastProps = dialogSpy.mock.calls.at(-1)?.[0];
    expect(lastProps.resetsAt).toBe(future);
  });

  it("falls back to weekly resets_at when daily is missing", () => {
    const weeklyFuture = new Date(
      Date.now() + 3 * 24 * 60 * 60 * 1000,
    ).toISOString();
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: {
        daily: null,
        weekly: { percent_used: 100, resets_at: weeklyFuture },
      },
      isSuccess: true,
      isError: false,
    });
    setSubscription();

    render(
      <RateLimitGate rateLimitMessage="limit reached" onDismiss={vi.fn()} />,
    );

    const lastProps = dialogSpy.mock.calls.at(-1)?.[0];
    expect(lastProps.isOpen).toBe(true);
    expect(lastProps.resetsAt).toBe(weeklyFuture);
  });

  it("passes null resetsAt when neither daily nor weekly data exists", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: { daily: null, weekly: null },
      isSuccess: true,
      isError: false,
    });
    setSubscription();

    render(
      <RateLimitGate rateLimitMessage="limit reached" onDismiss={vi.fn()} />,
    );

    const lastProps = dialogSpy.mock.calls.at(-1)?.[0];
    expect(lastProps.isOpen).toBe(true);
    expect(lastProps.resetsAt).toBeNull();
  });

  it("prefers daily resets_at over weekly when both are present", () => {
    const dailyFuture = new Date(Date.now() + 4 * 60 * 60 * 1000).toISOString();
    const weeklyFuture = new Date(
      Date.now() + 3 * 24 * 60 * 60 * 1000,
    ).toISOString();
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: {
        daily: { percent_used: 100, resets_at: dailyFuture },
        weekly: { percent_used: 80, resets_at: weeklyFuture },
      },
      isSuccess: true,
      isError: false,
    });
    setSubscription();

    render(
      <RateLimitGate rateLimitMessage="limit reached" onDismiss={vi.fn()} />,
    );

    const lastProps = dialogSpy.mock.calls.at(-1)?.[0];
    expect(lastProps.resetsAt).toBe(dailyFuture);
  });
});

describe("RateLimitGate — continuing on a linked subscription", () => {
  const capFailure = {
    kind: "usage_limit" as const,
    message: "You've reached your daily usage limit. Resets in 1h 0m.",
    authProvider: "platform",
    credentialId: null,
    resetsAt: null,
    retryable: false,
    reconnectFixesIt: false,
  };

  function usageLoaded() {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: { daily: { percent_used: 100, resets_at: null }, weekly: null },
      isSuccess: true,
      isError: false,
    });
    setSubscription("MAX");
  }

  it("asks for an alternative only when the cap came with an envelope", () => {
    usageLoaded();

    render(
      <RateLimitGate
        rateLimitMessage="limit reached"
        sessionId="sess-1"
        onDismiss={vi.fn()}
      />,
    );

    // A bare-string 429 -- an older backend, or a 429 that is not the usage
    // cap -- has no envelope to name the failed connection with, so there is
    // nothing to exclude and nothing to offer. The hook is told so.
    expect(mockUseProviderLimitDialog).toHaveBeenCalledWith(
      expect.objectContaining({ failure: null, sessionId: "sess-1" }),
    );
    const lastProps = dialogSpy.mock.calls.at(-1)?.[0];
    expect(lastProps.alternative).toBeNull();
  });

  it("hands the envelope to the switch hook and the offer to the dialog", () => {
    usageLoaded();
    const continueHere = vi.fn();
    mockUseProviderLimitDialog.mockReturnValue({
      alternative: {
        display_name: "ChatGPT",
        auth_provider: "codex",
        credential_id: "cred-1",
      },
      continueHere,
      isSwitching: true,
      isLoadingOffers: false,
      failedToLoadOffers: false,
      retryOffers: vi.fn(),
      resetHint: null,
    });
    const onDismiss = vi.fn();

    render(
      <RateLimitGate
        rateLimitMessage={capFailure.message}
        failure={capFailure}
        sessionId="sess-1"
        onDismiss={onDismiss}
      />,
    );

    expect(mockUseProviderLimitDialog).toHaveBeenCalledWith({
      failure: capFailure,
      sessionId: "sess-1",
      onDismiss,
    });
    const lastProps = dialogSpy.mock.calls.at(-1)?.[0];
    expect(lastProps.isOpen).toBe(true);
    expect(lastProps.alternative?.display_name).toBe("ChatGPT");
    expect(lastProps.onContinue).toBe(continueHere);
    expect(lastProps.isSwitching).toBe(true);
    // The upgrade path is untouched by the offer: the tier still reaches the
    // dialog so it can keep its Upgrade / Contact us CTA beside the switch.
    expect(lastProps.tier).toBe("MAX");
  });
});
