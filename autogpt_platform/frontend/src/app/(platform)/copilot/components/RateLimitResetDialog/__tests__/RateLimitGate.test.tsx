import {
  cleanup,
  render,
  screen,
  fireEvent,
  act,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getUsageExperience } from "@/services/usageExperience/helpers";
const actions = vi.hoisted(() => ({ use: vi.fn() }));
const provider = vi.hoisted(() => ({ use: vi.fn() }));
vi.mock("@/services/usageExperience/useUsageActions", () => ({
  useUsageActions: actions.use,
}));
vi.mock("../../ProviderLimitDialog/useProviderLimitDialog", () => ({
  useProviderLimitDialog: provider.use,
}));
import { RateLimitGate } from "../RateLimitGate";
const retry = vi.fn();
const upgrade = vi.fn();
const continueHere = vi.fn();
beforeEach(() => {
  actions.use.mockReturnValue({
    experience: getUsageExperience({
      tier: "PRO",
      daily: { percent_used: 100, resets_at: "2099-10-07T00:00:00Z" },
    }),
    isLoading: false,
    isError: false,
    retry,
    upgrade,
    isBillingEnabled: true,
    offer: { tier: "MAX" },
  });
  provider.use.mockReturnValue({
    alternative: { display_name: "ChatGPT" },
    continueHere,
    isSwitching: false,
  });
});
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});
describe("RateLimitGate", () => {
  it("opens on a backend refusal and preserves the linked-provider action", () => {
    render(
      <RateLimitGate
        rateLimitMessage="limit"
        onDismiss={vi.fn()}
        sessionId="chat"
      />,
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Continue on ChatGPT" }),
    );
    expect(continueHere).toHaveBeenCalledOnce();
  });
  it("shows a retry state instead of dismissing the refusal on usage API failure", async () => {
    actions.use.mockReturnValue({ ...actions.use(), isError: true });
    const dismiss = vi.fn();
    render(<RateLimitGate rateLimitMessage="limit" onDismiss={dismiss} />);
    fireEvent.click(await screen.findByRole("button", { name: "Try again" }));
    expect(retry.mock.calls.length).toBeGreaterThanOrEqual(2);
    expect(dismiss).not.toHaveBeenCalled();
  });
  it("refreshes a just-refused turn before trusting cached usage", async () => {
    let resolve: (() => void) | undefined;
    retry.mockReturnValueOnce(
      new Promise<void>((done) => {
        resolve = done;
      }),
    );
    const onRefreshingChange = vi.fn();
    render(
      <RateLimitGate
        rateLimitMessage="Weekly limit reached"
        onDismiss={vi.fn()}
        onRefreshingChange={onRefreshingChange}
      />,
    );
    expect(screen.getByText("Refreshing your plan and usage…")).toBeDefined();
    expect(
      screen.queryByRole("button", { name: "Review Max upgrade" }),
    ).toBeNull();
    expect(onRefreshingChange).toHaveBeenCalledWith(true);
    await act(async () => resolve?.());
    expect(
      await screen.findByRole("button", { name: "Review Max upgrade" }),
    ).toBeDefined();
    expect(onRefreshingChange).toHaveBeenLastCalledWith(false);
  });
  it("does not show the dialog without a backend refusal", () => {
    render(<RateLimitGate rateLimitMessage={null} onDismiss={vi.fn()} />);
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
