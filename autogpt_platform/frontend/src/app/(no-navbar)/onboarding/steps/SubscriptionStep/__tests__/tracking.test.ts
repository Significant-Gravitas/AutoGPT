import { beforeEach, describe, expect, it, vi } from "vitest";

const { sendDatafastEvent } = vi.hoisted(() => ({
  sendDatafastEvent: vi.fn(),
}));

vi.mock("@/services/analytics", () => ({
  analytics: { sendDatafastEvent },
}));

import { trackPaywallView } from "../tracking";

describe("trackPaywallView", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    sendDatafastEvent.mockReset();
    sessionStorage.clear();
  });

  it("reports the impression tagged with the pricing variant", () => {
    trackPaywallView("control");

    expect(sendDatafastEvent).toHaveBeenCalledTimes(1);
    expect(sendDatafastEvent).toHaveBeenCalledWith("paywall_view", {
      variant: "control",
    });
  });

  // Cancelling Stripe checkout returns to `?step=1&subscription=cancelled` as a
  // full navigation, remounting the paywall. Without the session guard that
  // second mount would inflate the funnel's denominator.
  it("reports at most once per session", () => {
    trackPaywallView("monthly-pro");
    trackPaywallView("monthly-pro");

    expect(sendDatafastEvent).toHaveBeenCalledTimes(1);
  });

  it("still reports when sessionStorage is unavailable", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("storage blocked");
    });

    trackPaywallView("yearly-max");

    expect(sendDatafastEvent).toHaveBeenCalledTimes(1);
  });

  // This runs in a mount effect on the screen users pay from: if a third-party
  // script failure escaped, React would unmount the paywall into an error
  // boundary and there would be nothing to buy.
  it("never throws when the DataFast script does", () => {
    sendDatafastEvent.mockImplementation(() => {
      throw new Error("datafast unavailable");
    });

    expect(() => trackPaywallView("control")).not.toThrow();
  });
});
