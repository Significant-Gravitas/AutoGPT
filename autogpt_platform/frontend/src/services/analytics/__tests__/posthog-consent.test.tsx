import { server } from "@/mocks/mock-server";
import {
  answerCookiebot,
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";
import { act, render } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { PostHog } from "posthog-js";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Runs the real posthog-js client against a stubbed PostHog host, so these
// assertions cover what the SDK itself stores and sends, not our calls to it.

const POSTHOG_HOST = "https://posthog.test";
const POSTHOG_KEY = "phc_test";
const EVENT_PATHS = ["/e/", "/i/v0/e/", "/batch/", "/capture/", "/s/"];

const FLAGS_RESPONSE = {
  flags: {
    "pricing-test": {
      key: "pricing-test",
      enabled: true,
      variant: "yearly",
      metadata: { id: 1, version: 1 },
    },
  },
};

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "*",
  "Access-Control-Allow-Methods": "*",
};

let requests: string[] = [];
let captured: string[] = [];

async function renderPostHog() {
  // A fresh client per test: init() is a no-op on one that already ran.
  vi.resetModules();
  const client = new PostHog();
  vi.doMock("posthog-js", () => ({
    default: client,
    posthog: client,
    PostHog,
  }));
  const [provider, { trackTabIntro }] = await Promise.all([
    import("@/providers/posthog/posthog-provider"),
    import("@/services/onboarding/tab-intro-analytics"),
  ]);
  const startSessionRecording = vi.spyOn(client, "startSessionRecording");
  const { PostHogProvider, PostHogPageViewTracker } = provider;
  render(
    <PostHogProvider>
      <PostHogPageViewTracker />
    </PostHogProvider>,
  );
  client.on("eventCaptured", (event) => {
    captured.push(event.event);
  });
  return { posthog: client, trackTabIntro, startSessionRecording };
}

function sentEvents() {
  return requests.filter((path) =>
    EVENT_PATHS.some((prefix) => path.startsWith(prefix)),
  );
}

function storedPostHogKeys() {
  const storage = [window.localStorage, window.sessionStorage].flatMap(
    (store) => Object.keys(store),
  );
  return [...storage, ...cookieNames()].filter(
    (key) => !key.startsWith("agpt_") && key !== "CookieConsent",
  );
}

function cookieNames() {
  return document.cookie
    .split(";")
    .map((part) => part.trim().split("=")[0])
    .filter(Boolean);
}

beforeEach(() => {
  vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "CLOUD");
  vi.stubEnv("NEXT_PUBLIC_POSTHOG_KEY", POSTHOG_KEY);
  vi.stubEnv("NEXT_PUBLIC_POSTHOG_HOST", POSTHOG_HOST);
  vi.stubEnv("NEXT_PUBLIC_COOKIEBOT_CBID", "");
  // posthog-js tries a script tag for remote config first; happy-dom refuses
  // to load scripts and reports it, before the SDK falls back to fetch.
  vi.spyOn(console, "error").mockImplementation(() => {});
  // PostHog drops events from browsers it takes for bots, HappyDOM included.
  vi.spyOn(navigator, "webdriver", "get").mockReturnValue(false);
  requests = [];
  captured = [];
  server.use(
    http.options(
      `${POSTHOG_HOST}/*`,
      () => new HttpResponse(null, { status: 204, headers: CORS_HEADERS }),
    ),
    http.all(`${POSTHOG_HOST}/*`, ({ request }) => {
      const path = new URL(request.url).pathname;
      requests.push(path);
      return HttpResponse.json(path === "/flags/" ? FLAGS_RESPONSE : {}, {
        headers: CORS_HEADERS,
      });
    }),
  );
});

afterEach(() => {
  removeCookiebot();
  window.localStorage.clear();
  window.sessionStorage.clear();
  cookieNames().forEach((name) => {
    document.cookie = `${name}=; Path=/; Max-Age=0`;
  });
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
});

describe("PostHog before analytics consent", () => {
  beforeEach(() => {
    configureCookiebot();
    installCookiebot();
  });

  it("captures nothing, stores nothing and records nothing", async () => {
    const { posthog, trackTabIntro, startSessionRecording } =
      await renderPostHog();

    posthog.capture("custom_event");
    trackTabIntro("tab_intro_shown", { tab: "agents" });

    expect(posthog.has_opted_out_capturing()).toBe(true);
    expect(posthog.config.disable_session_recording).toBe(true);
    expect(startSessionRecording).not.toHaveBeenCalled();
    expect(captured).toEqual([]);
    expect(storedPostHogKeys()).toEqual([]);
    await vi.waitFor(() => expect(requests).toContain("/flags/"));
    expect(sentEvents()).toEqual([]);
  });

  it("still evaluates feature flags for visitors who decline", async () => {
    const { posthog } = await renderPostHog();
    act(() => answerCookiebot({}));

    await vi.waitFor(() =>
      expect(posthog.getFeatureFlag("pricing-test")).toBe("yearly"),
    );
    expect(captured).toEqual([]);
    expect(storedPostHogKeys()).toEqual([]);
  });

  it("ignores an opt-in PostHog stored before consent was withdrawn", async () => {
    window.localStorage.setItem("agpt_posthog_consent", "1");

    const { posthog } = await renderPostHog();
    posthog.capture("custom_event");

    expect(posthog.has_opted_out_capturing()).toBe(true);
    expect(captured).toEqual([]);
    expect(storedPostHogKeys()).toEqual([]);
  });
});

describe("PostHog when analytics consent is granted", () => {
  beforeEach(() => {
    configureCookiebot();
  });

  it("opts in, persists and starts replay without a reload", async () => {
    installCookiebot();
    const { posthog, startSessionRecording } = await renderPostHog();
    expect(posthog.has_opted_out_capturing()).toBe(true);

    act(() => answerCookiebot({ statistics: true }));
    posthog.capture("custom_event");

    expect(posthog.has_opted_in_capturing()).toBe(true);
    expect(startSessionRecording).toHaveBeenCalledOnce();
    expect(posthog.config.disable_session_recording).toBe(false);
    expect(posthog.config.persistence).toBe("localStorage+cookie");
    expect(window.localStorage.getItem(`ph_${POSTHOG_KEY}_posthog`)).not.toBe(
      null,
    );
    expect(captured).toEqual(["$pageview", "custom_event"]);
  });

  it("drops events captured before consent instead of sending them later", async () => {
    installCookiebot();
    const { posthog, trackTabIntro } = await renderPostHog();
    posthog.capture("before_consent");
    trackTabIntro("tab_intro_shown", { tab: "agents" });

    act(() => answerCookiebot({ statistics: true }));
    posthog.capture("after_consent");

    expect(captured).toEqual(["$pageview", "after_consent"]);
  });

  it("starts opted in for a visitor who consented earlier", async () => {
    installCookiebot({ statistics: true });
    const { posthog, startSessionRecording } = await renderPostHog();

    posthog.capture("custom_event");

    expect(posthog.has_opted_in_capturing()).toBe(true);
    expect(startSessionRecording).toHaveBeenCalledOnce();
    expect(captured).toContain("custom_event");
  });

  it("opts back out when the visitor withdraws", async () => {
    installCookiebot({ statistics: true });
    vi.spyOn(window.location, "reload").mockImplementation(() => {});
    const { posthog } = await renderPostHog();

    act(() => answerCookiebot({}));
    posthog.capture("custom_event");

    expect(posthog.has_opted_out_capturing()).toBe(true);
    expect(captured).not.toContain("custom_event");
    expect(storedPostHogKeys()).toEqual([]);
  });
});

describe("PostHog without NEXT_PUBLIC_COOKIEBOT_CBID", () => {
  it("stays opted out even for a browser holding an old answer", async () => {
    installCookiebot({ statistics: true });
    const { posthog } = await renderPostHog();

    posthog.capture("custom_event");

    expect(posthog.has_opted_out_capturing()).toBe(true);
    expect(captured).toEqual([]);
    expect(storedPostHogKeys()).toEqual([]);
  });
});
