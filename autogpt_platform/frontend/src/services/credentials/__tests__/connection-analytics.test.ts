import { beforeEach, describe, expect, it, vi } from "vitest";
import { trackCredentialConnectionFailure } from "../connection-analytics";

const capture = vi.hoisted(() => vi.fn());
vi.mock("posthog-js", () => ({ default: { capture } }));

beforeEach(() => {
  vi.clearAllMocks();
});

describe("the failure class an alert rule groups on", () => {
  it("is the one the map assigns, not one a caller passes", () => {
    trackCredentialConnectionFailure("credential_oauth_popup_blocked", {
      provider: "github",
      failure_class: "class_99_whatever",
    });

    expect(capture).toHaveBeenCalledWith("credential_oauth_popup_blocked", {
      provider: "github",
      failure_class: "class_05_browser_channel_broken",
    });
  });

  it("never lets a blocked analytics host break a connect flow", () => {
    capture.mockImplementationOnce(() => {
      throw new Error("blocked by client");
    });

    expect(() =>
      trackCredentialConnectionFailure("credential_oauth_flow_timed_out"),
    ).not.toThrow();
  });
});
