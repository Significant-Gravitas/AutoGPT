import { ImpersonationState } from "@/lib/impersonation";
import {
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";
import type { CaptureResult } from "posthog-js";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { posthog } = vi.hoisted(() => ({
  posthog: {
    __loaded: true,
    get_session_id: vi.fn(() => "0192f0c4-5f4e-7b4c-9d6e-1a2b3c4d5e6f"),
  },
}));

vi.mock("posthog-js", () => ({ default: posthog }));

import {
  getPostHogSessionHeader,
  markImpersonatedEvent,
} from "../posthog-request-context";

const SESSION = "0192f0c4-5f4e-7b4c-9d6e-1a2b3c4d5e6f";

function event(): CaptureResult {
  return {
    uuid: "event-1",
    event: "paywall_viewed",
    properties: { surface: "billing" },
  } as CaptureResult;
}

beforeEach(() => {
  configureCookiebot();
  posthog.__loaded = true;
  ImpersonationState.clear();
});

afterEach(() => {
  ImpersonationState.clear();
  removeCookiebot();
  vi.unstubAllEnvs();
});

describe("markImpersonatedEvent", () => {
  it("marks every event an admin sends while acting as a user", () => {
    ImpersonationState.set("user-1");

    expect(markImpersonatedEvent(event())?.properties).toEqual({
      surface: "billing",
      impersonated: true,
      impersonated_user_id: "user-1",
    });
  });

  it("leaves everyone else's events as they are", () => {
    expect(markImpersonatedEvent(event())).toEqual(event());
    expect(markImpersonatedEvent(null)).toBeNull();
  });
});

describe("getPostHogSessionHeader", () => {
  it("names the PostHog session with analytics consent", () => {
    installCookiebot({ statistics: true });

    expect(getPostHogSessionHeader()).toEqual({
      "X-PostHog-Session-Id": SESSION,
    });
  });

  it("sends nothing without analytics consent", () => {
    installCookiebot({ statistics: false });

    expect(getPostHogSessionHeader()).toEqual({});
  });

  it("sends nothing while an admin acts as a user", () => {
    installCookiebot({ statistics: true });
    ImpersonationState.set("user-1");

    expect(getPostHogSessionHeader()).toEqual({});
  });

  it("sends nothing before PostHog has loaded", () => {
    installCookiebot({ statistics: true });
    posthog.__loaded = false;

    expect(getPostHogSessionHeader()).toEqual({});
  });
});
