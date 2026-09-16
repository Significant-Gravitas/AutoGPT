import { beforeEach, describe, expect, it, vi } from "vitest";
import { trackFunnel } from "./experts-analytics";

const { postAnalyticsMock } = vi.hoisted(() => ({
  postAnalyticsMock: vi.fn(),
}));

vi.mock("@/app/api/__generated__/endpoints/analytics/analytics", () => ({
  postAnalyticsLogRawAnalytics: postAnalyticsMock,
}));

describe("trackFunnel", () => {
  beforeEach(() => {
    postAnalyticsMock.mockReset();
  });

  it("posts the event, its payload and the analytics index", async () => {
    postAnalyticsMock.mockResolvedValueOnce({ status: "ok" });

    trackFunnel("expert_profile_opened", { template_id: "template-maria" });

    await vi.waitFor(() => expect(postAnalyticsMock).toHaveBeenCalledOnce());
    expect(postAnalyticsMock).toHaveBeenCalledWith({
      type: "expert_profile_opened",
      data: { template_id: "template-maria" },
      data_index: "expert_profile_opened",
    });
  });

  it("posts an empty payload for a view event", async () => {
    postAnalyticsMock.mockResolvedValueOnce({ status: "ok" });

    trackFunnel("experts_section_viewed");

    await vi.waitFor(() => expect(postAnalyticsMock).toHaveBeenCalledOnce());
    expect(postAnalyticsMock).toHaveBeenCalledWith({
      type: "experts_section_viewed",
      data: {},
      data_index: "experts_section_viewed",
    });
  });

  it("swallows a rejected analytics request", async () => {
    postAnalyticsMock.mockRejectedValueOnce(new Error("analytics unavailable"));

    expect(() => trackFunnel("home_viewed")).not.toThrow();
    await vi.waitFor(() => expect(postAnalyticsMock).toHaveBeenCalledOnce());
  });
});
