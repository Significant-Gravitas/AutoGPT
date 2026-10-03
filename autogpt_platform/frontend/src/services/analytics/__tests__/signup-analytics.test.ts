import { beforeEach, describe, expect, it, vi } from "vitest";

const capture = vi.hoisted(() => vi.fn());
vi.mock("posthog-js", () => ({ default: { capture } }));

import { trackSignupMarketingOptOut } from "../signup-analytics";

beforeEach(() => {
  capture.mockReset();
});

describe("trackSignupMarketingOptOut", () => {
  it("reports the opt-out with no properties", () => {
    trackSignupMarketingOptOut();

    expect(capture.mock.calls).toEqual([["signup_marketing_opt_out"]]);
  });

  it("swallows a blocked analytics host rather than breaking signup", () => {
    capture.mockImplementation(() => {
      throw new Error("blocked by client");
    });

    expect(() => trackSignupMarketingOptOut()).not.toThrow();
  });
});
