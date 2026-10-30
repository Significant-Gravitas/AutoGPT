import {
  ImpersonationState,
  setImpersonationViewer,
} from "@/lib/impersonation";
import {
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";
import type { CaptureResult, PostHog } from "posthog-js";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

type SessionRecordingStatus = NonNullable<
  PostHog["sessionRecording"]
>["status"];

const { posthog } = vi.hoisted(() => ({
  posthog: {
    __loaded: true,
    get_session_id: vi.fn(() => "0192f0c4-5f4e-7b4c-9d6e-1a2b3c4d5e6f"),
    sessionRecording: { status: "active" } as
      | { status: SessionRecordingStatus }
      | undefined,
    get_session_replay_url: vi.fn(
      () => "https://eu.posthog.com/project/1/replay/session-1?t=10",
    ),
  },
}));

vi.mock("posthog-js", () => ({ default: posthog }));

import {
  getPostHogReplayUrl,
  getPostHogSessionHeader,
  markImpersonatedEvent,
} from "../posthog-request-context";

const SESSION = "0192f0c4-5f4e-7b4c-9d6e-1a2b3c4d5e6f";
const ADMIN = { id: "admin-1", role: "admin" };

// What the auth store does once the admin who chose the target signs in.
function actAs(userID: string) {
  ImpersonationState.set(userID, ADMIN.id);
  setImpersonationViewer(ADMIN);
}

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
  posthog.sessionRecording = { status: "active" };
  ImpersonationState.clear();
});

afterEach(() => {
  ImpersonationState.clear();
  setImpersonationViewer(null);
  removeCookiebot();
  vi.unstubAllEnvs();
});

describe("markImpersonatedEvent", () => {
  it("marks every event an admin sends while acting as a user", () => {
    actAs("user-1");

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
    actAs("user-1");

    expect(getPostHogSessionHeader()).toEqual({});
  });

  it("sends nothing before PostHog has loaded", () => {
    installCookiebot({ statistics: true });
    posthog.__loaded = false;

    expect(getPostHogSessionHeader()).toEqual({});
  });
});

describe("getPostHogReplayUrl", () => {
  const REPLAY = "https://eu.posthog.com/project/1/replay/session-1?t=10";

  it("links the recorded session from just before now, with analytics consent", () => {
    installCookiebot({ statistics: true });

    expect(getPostHogReplayUrl()).toBe(REPLAY);
    expect(posthog.get_session_replay_url).toHaveBeenCalledWith({
      withTimestamp: true,
      timestampLookBack: 30,
    });
  });

  it("gives nothing without analytics consent", () => {
    installCookiebot({ statistics: false });

    expect(getPostHogReplayUrl()).toBe("");
  });

  it("gives nothing while an admin acts as a user", () => {
    installCookiebot({ statistics: true });
    actAs("user-1");

    expect(getPostHogReplayUrl()).toBe("");
  });

  it("gives nothing while a stored impersonation waits for the viewer to load", () => {
    installCookiebot({ statistics: true });
    ImpersonationState.set("user-1", ADMIN.id);
    setImpersonationViewer(null);

    expect(ImpersonationState.get()).toBeNull();
    expect(getPostHogReplayUrl()).toBe("");
  });

  it("links a sampled-in session too", () => {
    installCookiebot({ statistics: true });
    posthog.sessionRecording = { status: "sampled" };

    expect(getPostHogReplayUrl()).toBe(REPLAY);
  });

  it.each<SessionRecordingStatus>([
    "disabled",
    "buffering",
    "paused",
    "lazy_loading",
  ])(
    "gives nothing while the recording is %s, since nothing is uploaded",
    (status) => {
      installCookiebot({ statistics: true });
      posthog.sessionRecording = { status };

      expect(getPostHogReplayUrl()).toBe("");
    },
  );

  it("gives nothing before session recording has loaded", () => {
    installCookiebot({ statistics: true });
    posthog.sessionRecording = undefined;

    expect(getPostHogReplayUrl()).toBe("");
  });
});
