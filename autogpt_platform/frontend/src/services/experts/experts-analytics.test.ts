import { beforeEach, describe, expect, it, vi } from "vitest";
import { trackFunnel } from "./experts-analytics";

const { capture, addBreadcrumb } = vi.hoisted(() => ({
  capture: vi.fn(),
  addBreadcrumb: vi.fn(),
}));

vi.mock("posthog-js", () => ({ default: { capture } }));
vi.mock("@sentry/nextjs", () => ({ addBreadcrumb }));

describe("trackFunnel", () => {
  beforeEach(() => {
    capture.mockReset();
    addBreadcrumb.mockReset();
  });

  it("captures the event and its payload to PostHog", () => {
    trackFunnel("expert_profile_opened", { template_id: "template-maria" });

    expect(capture).toHaveBeenCalledExactlyOnceWith("expert_profile_opened", {
      template_id: "template-maria",
    });
  });

  it("captures a view event with no payload", () => {
    trackFunnel("experts_section_viewed");

    expect(capture).toHaveBeenCalledExactlyOnceWith(
      "experts_section_viewed",
      undefined,
    );
  });

  it("leaves a breadcrumb so the step shows on an error's timeline", () => {
    trackFunnel("hire_started", {
      template_id: "template-maria",
      surface: "expert_page",
    });

    expect(addBreadcrumb).toHaveBeenCalledExactlyOnceWith({
      category: "funnel",
      message: "hire_started",
      data: { template_id: "template-maria", surface: "expert_page" },
      level: "info",
    });
  });

  it("breadcrumbs before capturing, so a failing capture still leaves one", () => {
    capture.mockImplementation(() => {
      throw new Error("analytics unavailable");
    });

    expect(() => trackFunnel("home_viewed")).not.toThrow();
    expect(addBreadcrumb).toHaveBeenCalledOnce();
  });

  it("still captures when the breadcrumb throws", () => {
    // The two sinks are independent: losing Sentry must not lose PostHog.
    addBreadcrumb.mockImplementation(() => {
      throw new Error("sentry unavailable");
    });

    expect(() => trackFunnel("briefing_opened")).not.toThrow();
    expect(capture).toHaveBeenCalledExactlyOnceWith(
      "briefing_opened",
      undefined,
    );
  });
});
