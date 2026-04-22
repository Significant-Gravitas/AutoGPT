import { cleanup, render } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";

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

vi.mock("@/hooks/useCredits", () => ({
  default: () => ({ credits: 1000, fetchCredits: vi.fn() }),
}));

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: { ENABLE_PLATFORM_PAYMENT: "ENABLE_PLATFORM_PAYMENT" },
  useGetFlag: () => false,
}));

// Capture props the dialog was rendered with so we can assert on them.
const dialogSpy = vi.fn();
vi.mock("../RateLimitResetDialog", () => ({
  RateLimitResetDialog: (props: { isOpen: boolean }) => {
    dialogSpy(props);
    return <div data-testid="reset-dialog" data-open={String(props.isOpen)} />;
  },
}));

import { RateLimitGate } from "../RateLimitGate";

afterEach(() => {
  cleanup();
  mockToast.mockReset();
  mockUseGetV2GetCopilotUsage.mockReset();
  dialogSpy.mockReset();
});

describe("RateLimitGate", () => {
  it("disables the usage query when no rate-limit message is present", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: undefined,
      isSuccess: false,
      isError: false,
    });

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

  it("opens the reset dialog when usage has a positive reset cost", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: { reset_cost: 50, weekly: { percent_used: 100 } },
      isSuccess: true,
      isError: false,
    });

    render(
      <RateLimitGate rateLimitMessage="limit reached" onDismiss={vi.fn()} />,
    );

    expect(dialogSpy).toHaveBeenCalled();
    const lastProps = dialogSpy.mock.calls.at(-1)?.[0];
    expect(lastProps.isOpen).toBe(true);
    expect(mockToast).not.toHaveBeenCalled();
  });

  it("falls back to a toast when usage query errors", () => {
    const onDismiss = vi.fn();
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: undefined,
      isSuccess: false,
      isError: true,
    });

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

    // Dialog stays closed when the fallback fired.
    const lastProps = dialogSpy.mock.calls.at(-1)?.[0];
    expect(lastProps.isOpen).toBe(false);
  });

  it("falls back to a toast when reset cost is non-positive (credit reset unavailable)", () => {
    const onDismiss = vi.fn();
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: { reset_cost: 0, weekly: { percent_used: 100 } },
      isSuccess: true,
      isError: false,
    });

    render(
      <RateLimitGate rateLimitMessage="limit reached" onDismiss={onDismiss} />,
    );

    expect(mockToast).toHaveBeenCalledTimes(1);
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

    render(<RateLimitGate rateLimitMessage={null} onDismiss={vi.fn()} />);

    expect(mockToast).not.toHaveBeenCalled();
  });
});
