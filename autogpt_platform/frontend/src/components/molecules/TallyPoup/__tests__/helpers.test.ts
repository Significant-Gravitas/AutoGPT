import { beforeEach, describe, expect, it, vi } from "vitest";

const { getPostHogReplayUrl } = vi.hoisted(() => ({
  getPostHogReplayUrl: vi.fn(() => ""),
}));

vi.mock("@/services/analytics/posthog-request-context", () => ({
  getPostHogReplayUrl,
}));

import { getFeedbackHiddenFields } from "../helpers";

const PAGE = { pageUrl: "https://platform.agpt.co/library?x=1" };

beforeEach(() => {
  getPostHogReplayUrl.mockReturnValue("");
});

describe("getFeedbackHiddenFields", () => {
  it("links the feedback to the person's PostHog replay", () => {
    getPostHogReplayUrl.mockReturnValue(
      "https://eu.posthog.com/project/1/replay/session-1?t=10",
    );

    expect(getFeedbackHiddenFields({ ...PAGE, isAuthenticated: true })).toEqual(
      {
        posthogReplayUrl:
          "https://eu.posthog.com/project/1/replay/session-1?t=10",
        pageUrl: "https://platform.agpt.co/library",
        isAuthenticated: "true",
      },
    );
  });

  it("leaves the replay out when there is none", () => {
    expect(getFeedbackHiddenFields({ ...PAGE, isAuthenticated: null })).toEqual(
      {
        pageUrl: "https://platform.agpt.co/library",
        isAuthenticated: "unknown",
      },
    );
  });
});
