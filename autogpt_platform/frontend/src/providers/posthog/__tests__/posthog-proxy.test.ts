import { describe, expect, it } from "vitest";
import { getPostHogProxyTarget } from "../posthog-proxy";

describe("getPostHogProxyTarget", () => {
  it("targets the PostHog Cloud region the host names", () => {
    expect(
      getPostHogProxyTarget({
        key: "phc_test",
        host: "https://eu.i.posthog.com",
      }),
    ).toEqual({
      apiHost: "https://eu.i.posthog.com",
      assetsHost: "https://eu-assets.i.posthog.com",
      uiHost: "https://eu.posthog.com",
    });
    expect(
      getPostHogProxyTarget({
        key: "phc_test",
        host: "https://us.i.posthog.com/",
      }),
    ).toEqual({
      apiHost: "https://us.i.posthog.com",
      assetsHost: "https://us-assets.i.posthog.com",
      uiHost: "https://us.posthog.com",
    });
  });

  it("accepts the older region host without the .i", () => {
    expect(
      getPostHogProxyTarget({ key: "phc_test", host: "https://eu.posthog.com" })
        ?.apiHost,
    ).toBe("https://eu.i.posthog.com");
  });

  it("leaves self-hosted, unset or keyless PostHog alone", () => {
    expect(
      getPostHogProxyTarget({
        key: "phc_test",
        host: "https://posthog.example.com",
      }),
    ).toBeNull();
    expect(getPostHogProxyTarget({ key: "phc_test", host: "" })).toBeNull();
    expect(
      getPostHogProxyTarget({ key: "", host: "https://eu.i.posthog.com" }),
    ).toBeNull();
  });
});
