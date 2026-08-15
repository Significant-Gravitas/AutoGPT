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

  it("swallows a rejected analytics request", async () => {
    postAnalyticsMock.mockRejectedValueOnce(new Error("analytics unavailable"));

    expect(() => trackFunnel("home_viewed")).not.toThrow();
    await vi.waitFor(() => expect(postAnalyticsMock).toHaveBeenCalledOnce());
  });
});
