import { beforeEach, describe, expect, it, vi } from "vitest";
import { buildAttributionPayload } from "../attribution-payload";

const anonymous = vi.hoisted(() => ({
  id: "anon-1" as string | null,
  deviceID: "device-1" as string | null,
  landing: null as null | {
    path: string;
    referrer: string | null;
    utm_source: string | null;
    utm_medium: string | null;
    utm_campaign: string | null;
    at: string;
  },
  signupMethod: null as string | null,
}));

vi.mock("@/services/analytics/anonymous-id", () => ({
  getAnonymousID: () => anonymous.id,
  getPostHogDeviceID: () => anonymous.deviceID,
  readFirstLanding: () => anonymous.landing,
}));

vi.mock("@/services/analytics/account-created-cookie", () => ({
  readAccountCreatedFlag: () => anonymous.signupMethod,
}));

beforeEach(() => {
  anonymous.id = "anon-1";
  anonymous.deviceID = "device-1";
  anonymous.landing = null;
  anonymous.signupMethod = null;
});

describe("buildAttributionPayload", () => {
  it("sends nulls for everything the browser does not know", () => {
    expect(buildAttributionPayload()).toEqual({
      anonymous_id: "anon-1",
      posthog_distinct_id: "device-1",
      landing_path: null,
      referrer: null,
      utm_source: null,
      utm_medium: null,
      utm_campaign: null,
      signup_method: null,
    });
  });

  it("flattens the first landing and the signup method into the payload", () => {
    anonymous.landing = {
      path: "/pricing?utm_source=x",
      referrer: "https://example.com/",
      utm_source: "x",
      utm_medium: "cpc",
      utm_campaign: "launch",
      at: "2026-09-04T00:00:00.000Z",
    };
    anonymous.signupMethod = "google";

    expect(buildAttributionPayload()).toMatchObject({
      landing_path: "/pricing?utm_source=x",
      referrer: "https://example.com/",
      utm_source: "x",
      utm_medium: "cpc",
      utm_campaign: "launch",
      signup_method: "google",
    });
  });
});
