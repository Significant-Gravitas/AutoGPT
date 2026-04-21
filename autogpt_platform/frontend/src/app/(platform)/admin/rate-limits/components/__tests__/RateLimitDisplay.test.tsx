import {
  render,
  screen,
  fireEvent,
  waitFor,
  cleanup,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RateLimitDisplay } from "../RateLimitDisplay";
import type { UserRateLimitResponse } from "@/app/api/__generated__/models/userRateLimitResponse";

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: vi.fn() }),
}));

const mockConfirm = vi.fn();

beforeEach(() => {
  mockConfirm.mockReset();
  window.confirm = mockConfirm;
});

afterEach(() => {
  cleanup();
});

function makeData(
  overrides: Partial<UserRateLimitResponse> = {},
): UserRateLimitResponse {
  return {
    user_id: "user-abc-123",
    user_email: "alice@example.com",
    daily_cost_limit_microdollars: 10_000_000,
    weekly_cost_limit_microdollars: 50_000_000,
    daily_cost_used_microdollars: 2_500_000,
    weekly_cost_used_microdollars: 10_000_000,
    tier: "FREE",
    ...overrides,
  };
}

describe("RateLimitDisplay", () => {
  it("renders the user email heading", () => {
    render(<RateLimitDisplay data={makeData()} onReset={vi.fn()} />);
    expect(
      screen.getByText(/Rate Limits for alice@example\.com/),
    ).toBeDefined();
  });

  it("renders user ID when email is present", () => {
    render(<RateLimitDisplay data={makeData()} onReset={vi.fn()} />);
    expect(screen.getByText(/user-abc-123/)).toBeDefined();
  });

  it("falls back to user_id in heading when email is absent", () => {
    render(
      <RateLimitDisplay
        data={makeData({ user_email: undefined })}
        onReset={vi.fn()}
      />,
    );
    expect(screen.getByText(/Rate Limits for user-abc-123/)).toBeDefined();
  });

  it("displays the current tier badge", () => {
    render(
      <RateLimitDisplay data={makeData({ tier: "PRO" })} onReset={vi.fn()} />,
    );
    const badge = screen.getByText("PRO");
    expect(badge).toBeDefined();
    expect(badge.className).toContain("bg-blue-100");
  });

  it("defaults unknown tier to FREE", () => {
    render(
      <RateLimitDisplay
        data={makeData({ tier: "UNKNOWN" as UserRateLimitResponse["tier"] })}
        onReset={vi.fn()}
      />,
    );
    const badge = screen.getByText("FREE");
    expect(badge).toBeDefined();
  });

  it("renders tier dropdown with all tiers", () => {
    render(<RateLimitDisplay data={makeData()} onReset={vi.fn()} />);
    const select = screen.getByLabelText("Subscription tier");
    expect(select).toBeDefined();
    expect(select.querySelectorAll("option").length).toBe(4);
  });

  it("disables tier dropdown when onTierChange is not provided", () => {
    render(<RateLimitDisplay data={makeData()} onReset={vi.fn()} />);
    const select = screen.getByLabelText(
      "Subscription tier",
    ) as HTMLSelectElement;
    expect(select.disabled).toBe(true);
  });

  it("enables tier dropdown when onTierChange is provided", () => {
    render(
      <RateLimitDisplay
        data={makeData()}
        onReset={vi.fn()}
        onTierChange={vi.fn()}
      />,
    );
    const select = screen.getByLabelText(
      "Subscription tier",
    ) as HTMLSelectElement;
    expect(select.disabled).toBe(false);
  });

  it("renders daily and weekly usage sections", () => {
    render(<RateLimitDisplay data={makeData()} onReset={vi.fn()} />);
    expect(screen.getByText("Daily Spend")).toBeDefined();
    expect(screen.getByText("Weekly Spend")).toBeDefined();
  });

  it("renders reset scope dropdown and reset button", () => {
    render(<RateLimitDisplay data={makeData()} onReset={vi.fn()} />);
    expect(screen.getByLabelText("Reset scope")).toBeDefined();
    expect(screen.getByText("Reset Usage")).toBeDefined();
  });

  it("disables reset button when nothing to reset", () => {
    render(
      <RateLimitDisplay
        data={makeData({ daily_cost_used_microdollars: 0 })}
        onReset={vi.fn()}
      />,
    );
    const button = screen.getByText("Reset Usage").closest("button")!;
    expect(button.disabled).toBe(true);
  });

  it("enables reset button when there is usage to reset", () => {
    render(
      <RateLimitDisplay
        data={makeData({ daily_cost_used_microdollars: 100_000 })}
        onReset={vi.fn()}
      />,
    );
    const button = screen.getByText("Reset Usage").closest("button")!;
    expect(button.disabled).toBe(false);
  });

  it("calls onReset when reset button is clicked and confirmed", async () => {
    const onReset = vi.fn().mockResolvedValue(undefined);
    mockConfirm.mockReturnValue(true);

    render(<RateLimitDisplay data={makeData()} onReset={onReset} />);

    fireEvent.click(screen.getByText("Reset Usage"));

    await waitFor(() => {
      expect(onReset).toHaveBeenCalledWith(false);
    });
  });

  it("does not call onReset when confirm is cancelled", () => {
    const onReset = vi.fn();
    mockConfirm.mockReturnValue(false);

    render(<RateLimitDisplay data={makeData()} onReset={onReset} />);

    fireEvent.click(screen.getByText("Reset Usage"));
    expect(onReset).not.toHaveBeenCalled();
  });

  it("passes resetWeekly=true when 'both' is selected", async () => {
    const onReset = vi.fn().mockResolvedValue(undefined);
    mockConfirm.mockReturnValue(true);

    render(
      <RateLimitDisplay
        data={makeData({ weekly_cost_used_microdollars: 100_000 })}
        onReset={onReset}
      />,
    );

    fireEvent.change(screen.getByLabelText("Reset scope"), {
      target: { value: "both" },
    });
    fireEvent.click(screen.getByText("Reset Usage"));

    await waitFor(() => {
      expect(onReset).toHaveBeenCalledWith(true);
    });
  });

  it("calls onTierChange when tier is changed and confirmed", async () => {
    const onTierChange = vi.fn().mockResolvedValue(undefined);
    mockConfirm.mockReturnValue(true);

    render(
      <RateLimitDisplay
        data={makeData({ tier: "FREE" })}
        onReset={vi.fn()}
        onTierChange={onTierChange}
      />,
    );

    fireEvent.change(screen.getByLabelText("Subscription tier"), {
      target: { value: "PRO" },
    });

    await waitFor(() => {
      expect(onTierChange).toHaveBeenCalledWith("PRO");
    });
  });

  it("does not call onTierChange when selecting the same tier", () => {
    const onTierChange = vi.fn();

    render(
      <RateLimitDisplay
        data={makeData({ tier: "FREE" })}
        onReset={vi.fn()}
        onTierChange={onTierChange}
      />,
    );

    fireEvent.change(screen.getByLabelText("Subscription tier"), {
      target: { value: "FREE" },
    });

    expect(onTierChange).not.toHaveBeenCalled();
  });

  it("does not call onTierChange when confirm is cancelled", () => {
    const onTierChange = vi.fn();
    mockConfirm.mockReturnValue(false);

    render(
      <RateLimitDisplay
        data={makeData({ tier: "FREE" })}
        onReset={vi.fn()}
        onTierChange={onTierChange}
      />,
    );

    fireEvent.change(screen.getByLabelText("Subscription tier"), {
      target: { value: "PRO" },
    });

    expect(onTierChange).not.toHaveBeenCalled();
  });

  it("catches error when onTierChange rejects", async () => {
    const onTierChange = vi.fn().mockRejectedValue(new Error("fail"));
    mockConfirm.mockReturnValue(true);

    render(
      <RateLimitDisplay
        data={makeData({ tier: "FREE" })}
        onReset={vi.fn()}
        onTierChange={onTierChange}
      />,
    );

    fireEvent.change(screen.getByLabelText("Subscription tier"), {
      target: { value: "PRO" },
    });

    await waitFor(() => {
      expect(onTierChange).toHaveBeenCalledWith("PRO");
    });
  });

  it("applies custom className when provided", () => {
    const { container } = render(
      <RateLimitDisplay
        data={makeData()}
        onReset={vi.fn()}
        className="custom-class"
      />,
    );
    expect(container.firstElementChild?.className).toBe("custom-class");
  });
});
