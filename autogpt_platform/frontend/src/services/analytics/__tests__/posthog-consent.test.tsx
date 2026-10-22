import type { User } from "@/lib/auth/types";
import { server } from "@/mocks/mock-server";
import {
  answerCookiebot,
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";
import { act, render } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { PostHog, type CaptureResult } from "posthog-js";
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

const USER = {
  id: "user-1",
  email: "ada@example.com",
  user_metadata: { name: "Ada" },
} as unknown as User;

interface FlagsRequest {
  distinct_id?: string;
  $device_id?: string;
  $anon_distinct_id?: string;
  person_properties?: Record<string, unknown>;
}

let requests: string[] = [];
let flagsRequests: FlagsRequest[] = [];
let captured: string[] = [];
let capturedEvents: CaptureResult[] = [];
let clients: PostHog[] = [];

interface RenderOptions {
  user?: User | null;
  // Whether the auth store has reported the session yet.
  sessionKnown?: boolean;
}

// Each call is a fresh page load: a new client (init() is a no-op on one that
// already ran) and fresh modules, so in-memory ids start over too.
async function renderPostHog({
  user = null,
  sessionKnown = true,
}: RenderOptions = {}) {
  vi.resetModules();
  const client = new PostHog();
  clients.push(client);
  vi.doMock("posthog-js", () => ({
    default: client,
    posthog: client,
    PostHog,
  }));
  vi.doMock("@/lib/auth/hooks/useAuth", () => ({
    useAuth: () => ({ user, isUserLoading: false }),
  }));
  const [
    provider,
    { trackTabIntro },
    { resetAnalyticsIdentity },
    { setAnalyticsIdentitySignedIn },
    { buildAttributionPayload },
  ] = await Promise.all([
    import("@/providers/posthog/posthog-provider"),
    import("@/services/onboarding/tab-intro-analytics"),
    import("@/services/analytics/reset-identity"),
    import("@/services/analytics/anonymous-id"),
    import("@/services/analytics/attribution-payload"),
  ]);
  // What the auth store reports once the session is known.
  if (sessionKnown) setAnalyticsIdentitySignedIn(Boolean(user));
  const startSessionRecording = vi.spyOn(client, "startSessionRecording");
  const identify = vi.spyOn(client, "identify");
  const { PostHogProvider, PostHogPageViewTracker, PostHogUserTracker } =
    provider;
  render(
    <PostHogProvider>
      <PostHogUserTracker />
      <PostHogPageViewTracker />
    </PostHogProvider>,
  );
  client.on("eventCaptured", (event) => {
    captured.push(event.event);
    capturedEvents.push(event);
  });
  return {
    buildAttributionPayload,
    setAnalyticsIdentitySignedIn,
    posthog: client,
    trackTabIntro,
    startSessionRecording,
    identify,
    resetAnalyticsIdentity,
  };
}

function sentEvents() {
  return requests.filter((path) =>
    EVENT_PATHS.some((prefix) => path.startsWith(prefix)),
  );
}

// Everything this browser holds for analytics, our own agpt_ keys included.
function storedPostHogKeys() {
  const storage = [window.localStorage, window.sessionStorage].flatMap(
    (store) => Object.keys(store),
  );
  return [...storage, ...cookieNames()].filter(
    (key) => key !== "CookieConsent",
  );
}

function setPageURL(url: string) {
  const { happyDOM } = window as unknown as {
    happyDOM?: { setURL: (url: string) => void };
  };
  happyDOM?.setURL(url);
}

// Records every cookie write while still applying it.
function recordCookieWrites(): string[] {
  let proto: object | null = document;
  let descriptor: PropertyDescriptor | undefined;
  while (proto && !descriptor?.set) {
    descriptor = Object.getOwnPropertyDescriptor(proto, "cookie");
    proto = Object.getPrototypeOf(proto);
  }
  if (!descriptor?.get || !descriptor.set)
    throw new Error("no cookie accessor");
  const { get, set } = descriptor;
  const writes: string[] = [];
  Object.defineProperty(document, "cookie", {
    configurable: true,
    get: () => get.call(document),
    set: (value: string) => {
      writes.push(value);
      set.call(document, value);
    },
  });
  return writes;
}

async function readFlagsBody(request: Request): Promise<FlagsRequest> {
  const body = await request.text();
  const data = new URLSearchParams(body).get("data");
  const json = data ? Buffer.from(data, "base64").toString("utf8") : body;
  return JSON.parse(json) as FlagsRequest;
}

function seedEarlierVisit() {
  const earlier = JSON.stringify({
    distinct_id: "user-1",
    $device_id: "old-device",
  });
  window.localStorage.setItem("agpt_anonymous_id", "stored-id");
  window.localStorage.setItem("agpt_first_landing", '{"path":"/earlier"}');
  window.localStorage.setItem(`ph_${POSTHOG_KEY}_posthog`, earlier);
  window.sessionStorage.setItem(`ph_${POSTHOG_KEY}_window_id`, '"window"');
  window.sessionStorage.setItem(
    `ph_${POSTHOG_KEY}_primary_window_exists`,
    "true",
  );
  document.cookie = `ph_${POSTHOG_KEY}_posthog=${encodeURIComponent(earlier)}; Path=/`;
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
  flagsRequests = [];
  captured = [];
  capturedEvents = [];
  server.use(
    http.options(
      `${POSTHOG_HOST}/*`,
      () => new HttpResponse(null, { status: 204, headers: CORS_HEADERS }),
    ),
    http.all(`${POSTHOG_HOST}/*`, async ({ request }) => {
      const path = new URL(request.url).pathname;
      requests.push(path);
      if (path === "/flags/") flagsRequests.push(await readFlagsBody(request));
      return HttpResponse.json(path === "/flags/" ? FLAGS_RESPONSE : {}, {
        headers: CORS_HEADERS,
      });
    }),
  );
});

afterEach(() => {
  // A client from an earlier test would otherwise keep writing (and
  // deleting) the next test's cookies when a late response arrives.
  clients.forEach((client) => client.set_config({ persistence: "memory" }));
  clients = [];
  removeCookiebot();
  window.localStorage.clear();
  window.sessionStorage.clear();
  cookieNames().forEach((name) => {
    document.cookie = `${name}=; Path=/; Expires=Thu, 01 Jan 1970 00:00:00 GMT`;
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

  it("leaves what an earlier visit stored while the answer is pending", async () => {
    seedEarlierVisit();
    const stored = storedPostHogKeys().sort();

    await renderPostHog();

    expect(storedPostHogKeys().sort()).toEqual(stored);
  });

  it("deletes what an earlier visit stored once the visitor declines", async () => {
    seedEarlierVisit();
    await renderPostHog();

    act(() => answerCookiebot({}));

    expect(storedPostHogKeys()).toEqual([]);
  });

  it("sends /flags a fresh id for each page load, never a stored one", async () => {
    seedEarlierVisit();

    await renderPostHog();
    await vi.waitFor(() => expect(flagsRequests).toHaveLength(1));
    const [first] = flagsRequests;

    expect(first.distinct_id).toBeTruthy();
    expect(["stored-id", "user-1", "old-device"]).not.toContain(
      first.distinct_id,
    );
    expect(first.$device_id).toBe(first.distinct_id);
    expect(first.person_properties).toEqual({});

    seedEarlierVisit();
    await renderPostHog();
    await vi.waitFor(() => expect(flagsRequests).toHaveLength(2));

    expect(flagsRequests[1].distinct_id).not.toBe(first.distinct_id);
    expect(["stored-id", "user-1", "old-device"]).not.toContain(
      flagsRequests[1].distinct_id,
    );
  });

  it("holds identify until consent, then identifies on a late grant", async () => {
    const { posthog, identify } = await renderPostHog({ user: USER });
    await vi.waitFor(() => expect(flagsRequests).toHaveLength(1));

    expect(identify).not.toHaveBeenCalled();
    expect(posthog.get_distinct_id()).not.toBe(USER.id);
    expect(JSON.stringify(flagsRequests)).not.toContain(USER.id);
    expect(JSON.stringify(flagsRequests)).not.toContain("ada@example.com");

    act(() => answerCookiebot({ statistics: true }));

    expect(identify).toHaveBeenCalledOnce();
    expect(identify).toHaveBeenCalledWith(USER.id, {
      email: USER.email,
      name: "Ada",
    });
    expect(posthog.get_distinct_id()).toBe(USER.id);
    await vi.waitFor(() =>
      expect(flagsRequests.map((body) => body.distinct_id)).toContain(USER.id),
    );
  });

  it("does not identify a visitor who declines", async () => {
    const { identify } = await renderPostHog({ user: USER });

    act(() => answerCookiebot({}));
    await vi.waitFor(() => expect(flagsRequests.length).toBeGreaterThan(0));

    expect(identify).not.toHaveBeenCalled();
    expect(JSON.stringify(flagsRequests)).not.toContain(USER.id);
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
    const pageID = posthog.get_distinct_id();

    act(() => answerCookiebot({ statistics: true }));
    posthog.capture("custom_event");

    expect(posthog.has_opted_in_capturing()).toBe(true);
    expect(startSessionRecording).toHaveBeenCalledOnce();
    expect(posthog.config.disable_session_recording).toBe(false);
    expect(posthog.config.persistence).toBe("localStorage+cookie");
    expect(window.localStorage.getItem(`ph_${POSTHOG_KEY}_posthog`)).not.toBe(
      null,
    );
    expect(window.localStorage.getItem("agpt_anonymous_id")).toBe(pageID);
    expect(window.localStorage.getItem("agpt_first_landing")).not.toBeNull();
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

  it("opts back out and deletes its storage when the visitor withdraws", async () => {
    installCookiebot({ statistics: true });
    const { posthog } = await renderPostHog();
    posthog.capture("before_withdrawal");
    expect(captured).toContain("before_withdrawal");
    expect(storedPostHogKeys()).toEqual(
      expect.arrayContaining([
        `ph_${POSTHOG_KEY}_posthog`,
        "agpt_anonymous_id",
        "agpt_first_landing",
      ]),
    );

    act(() => answerCookiebot({}));
    posthog.capture("after_withdrawal");

    expect(posthog.has_opted_out_capturing()).toBe(true);
    expect(captured).not.toContain("after_withdrawal");
    expect(storedPostHogKeys()).toEqual([]);
  });

  it("keeps a consenting visitor opted in after logout", async () => {
    installCookiebot({ statistics: true });
    const { posthog, resetAnalyticsIdentity } = await renderPostHog({
      user: USER,
    });
    const before = posthog.get_distinct_id();

    resetAnalyticsIdentity();
    posthog.capture("after_logout");

    expect(posthog.get_distinct_id()).not.toBe(before);
    expect(posthog.has_opted_in_capturing()).toBe(true);
    expect(captured).toContain("after_logout");
    expect(window.localStorage.getItem("agpt_anonymous_id")).toBe(
      posthog.get_distinct_id(),
    );
  });

  it("still deletes its storage when the visitor withdraws after a logout", async () => {
    installCookiebot({ statistics: true });
    const { posthog, resetAnalyticsIdentity } = await renderPostHog({
      user: USER,
    });
    resetAnalyticsIdentity();
    posthog.capture("after_logout");
    expect(storedPostHogKeys()).toContain(`ph_${POSTHOG_KEY}_posthog`);

    act(() => answerCookiebot({}));

    expect(posthog.has_opted_out_capturing()).toBe(true);
    expect(storedPostHogKeys()).toEqual([]);
  });
});

describe("PostHog storage on a parent domain", () => {
  afterEach(() => {
    Reflect.deleteProperty(document, "cookie");
    setPageURL("http://localhost:3000/");
  });

  it("deletes the host cookie and every parent-domain variant", async () => {
    setPageURL("https://platform.agpt.co/");
    configureCookiebot();
    installCookiebot({});
    const { forgetPostHogStorageWithoutConsent } = await import(
      "@/providers/posthog/posthog-consent"
    );
    document.cookie = `ph_${POSTHOG_KEY}_posthog=x; Path=/; Domain=.agpt.co`;
    const writes = recordCookieWrites();

    forgetPostHogStorageWithoutConsent(POSTHOG_KEY);

    const deletions = writes.filter((value) =>
      value.startsWith(`ph_${POSTHOG_KEY}_posthog=;`),
    );
    expect(deletions).toEqual(
      expect.arrayContaining([
        expect.not.stringContaining("Domain="),
        expect.stringContaining("Domain=.platform.agpt.co"),
        expect.stringContaining("Domain=.agpt.co"),
      ]),
    );
    expect(cookieNames()).not.toContain(`ph_${POSTHOG_KEY}_posthog`);
  });
});

// The marketing site (agpt.co) runs PostHog on the same project and keeps its
// state in the ph_<key>_posthog cookie on .agpt.co. A visitor who comes over
// keeps that identity, so their website visits join the account they create.
describe("PostHog hand-over from the marketing site", () => {
  const PERSISTENCE = `ph_${POSTHOG_KEY}_posthog`;
  const WEBSITE_ID = "website-device";

  function seedWebsiteCookie() {
    const state = JSON.stringify({
      distinct_id: WEBSITE_ID,
      $device_id: WEBSITE_ID,
    });
    document.cookie = `${PERSISTENCE}=${encodeURIComponent(state)}; Path=/; Domain=.agpt.co`;
  }

  function sharedCookie(): Record<string, unknown> | null {
    const entry = document.cookie
      .split("; ")
      .find((part) => part.startsWith(`${PERSISTENCE}=`));
    if (!entry) return null;
    return JSON.parse(decodeURIComponent(entry.slice(PERSISTENCE.length + 1)));
  }

  beforeEach(() => {
    setPageURL("https://platform.agpt.co/");
    configureCookiebot();
    seedWebsiteCookie();
  });

  afterEach(() => {
    document.cookie = `${PERSISTENCE}=; Path=/; Domain=.agpt.co; Expires=Thu, 01 Jan 1970 00:00:00 GMT`;
    setPageURL("http://localhost:3000/");
  });

  it("keeps the website's cookie while the visitor has not answered", async () => {
    installCookiebot();

    const { posthog } = await renderPostHog();
    await vi.waitFor(() => expect(flagsRequests).toHaveLength(1));

    expect(sharedCookie()).toEqual({
      distinct_id: WEBSITE_ID,
      $device_id: WEBSITE_ID,
    });
    expect(posthog.get_distinct_id()).not.toBe(WEBSITE_ID);
    expect(flagsRequests[0].distinct_id).not.toBe(WEBSITE_ID);
  });

  it("deletes the website's cookie when the visitor declines", async () => {
    installCookiebot();
    await renderPostHog();

    act(() => answerCookiebot({}));

    expect(sharedCookie()).toBeNull();
  });

  it("moves onto the website's id when the visitor accepts", async () => {
    installCookiebot();
    const { posthog } = await renderPostHog();
    await vi.waitFor(() => expect(flagsRequests).toHaveLength(1));

    act(() => answerCookiebot({ statistics: true }));
    posthog.capture("after_consent");

    expect(posthog.get_distinct_id()).toBe(WEBSITE_ID);
    expect(posthog.get_property("$device_id")).toBe(WEBSITE_ID);
    expect(window.localStorage.getItem("agpt_anonymous_id")).toBe(WEBSITE_ID);
    expect(sharedCookie()).toMatchObject({
      distinct_id: WEBSITE_ID,
      $device_id: WEBSITE_ID,
    });
    expect(captured).toEqual(["$pageview", "after_consent"]);
    await vi.waitFor(() =>
      expect(flagsRequests.map((body) => body.distinct_id)).toContain(
        WEBSITE_ID,
      ),
    );
  });

  it("moves onto the website's id when Cookiebot grants after PostHog started", async () => {
    // Regions with implied consent: no stored answer yet, and uc.js arrives
    // after PostHog has started on an in-memory id.
    const { posthog } = await renderPostHog();

    act(() => {
      installCookiebot({ statistics: true });
      window.dispatchEvent(new Event("CookiebotOnConsentReady"));
    });

    expect(posthog.get_distinct_id()).toBe(WEBSITE_ID);
    expect(sharedCookie()).toMatchObject({ $device_id: WEBSITE_ID });
  });

  it("starts on the website's id for a visitor who allowed analytics before", async () => {
    installCookiebot({ statistics: true });

    const { posthog } = await renderPostHog();

    expect(posthog.get_distinct_id()).toBe(WEBSITE_ID);
    expect(window.localStorage.getItem("agpt_anonymous_id")).toBe(WEBSITE_ID);
  });

  it("merges the website visits into an account created here", async () => {
    document.cookie = "agpt_account_created=email; Path=/; Secure";
    installCookiebot();
    const { posthog, buildAttributionPayload } = await renderPostHog({
      user: USER,
    });

    act(() => answerCookiebot({ statistics: true }));

    const identify = capturedEvents.find(
      (event) => event.event === "$identify",
    );
    expect(identify?.properties.$anon_distinct_id).toBe(WEBSITE_ID);
    expect(posthog.get_distinct_id()).toBe(USER.id);
    expect(buildAttributionPayload()).toMatchObject({
      anonymous_id: WEBSITE_ID,
      signup_method: "email",
    });
    document.cookie = "agpt_account_created=; Path=/; Max-Age=0; Secure";
  });

  it("gives the backend the website's id once the visitor accepts", async () => {
    installCookiebot();
    const { buildAttributionPayload } = await renderPostHog();

    act(() => answerCookiebot({ statistics: true }));

    expect(buildAttributionPayload()).toMatchObject({
      anonymous_id: WEBSITE_ID,
      posthog_distinct_id: WEBSITE_ID,
    });
  });

  it("doesn't hand the website visits to an older account signed in here", async () => {
    installCookiebot();
    const { posthog } = await renderPostHog({ user: USER });
    const pageID = posthog.get_distinct_id();

    act(() => answerCookiebot({ statistics: true }));

    const identify = capturedEvents.find(
      (event) => event.event === "$identify",
    );
    expect(identify?.properties.$anon_distinct_id).toBe(pageID);
    expect(pageID).not.toBe(WEBSITE_ID);
    expect(window.localStorage.getItem("agpt_anonymous_id")).toBe(pageID);
    expect(sharedCookie()).not.toMatchObject({ $device_id: WEBSITE_ID });
  });

  it("forgets the website's id on logout while the answer is pending", async () => {
    installCookiebot();
    const { resetAnalyticsIdentity } = await renderPostHog({ user: USER });

    resetAnalyticsIdentity();

    expect(sharedCookie()).toBeNull();
  });

  it("keeps a returning visitor's own id", async () => {
    window.localStorage.setItem("agpt_anonymous_id", "app-id");
    installCookiebot();
    const { posthog } = await renderPostHog();

    act(() => answerCookiebot({ statistics: true }));

    expect(posthog.get_distinct_id()).toBe("app-id");
    expect(window.localStorage.getItem("agpt_anonymous_id")).toBe("app-id");
  });

  it("keeps the cookie and stops writing when Cookiebot asks again", async () => {
    installCookiebot({ statistics: true });
    const { posthog } = await renderPostHog();
    posthog.capture("before_asking");
    const writes = recordCookieWrites();

    act(() => {
      if (window.Cookiebot) window.Cookiebot.hasResponse = false;
      window.dispatchEvent(new Event("CookiebotOnDialogInit"));
      window.dispatchEvent(new Event("CookiebotOnLoad"));
    });
    const writesBeforeCapture = writes.length;
    posthog.capture("while_asking");

    expect(captured).toContain("before_asking");
    expect(captured).not.toContain("while_asking");
    expect(sharedCookie()).toMatchObject({
      distinct_id: WEBSITE_ID,
      $device_id: WEBSITE_ID,
    });
    expect(
      writes.filter((value) => value.includes("Expires=Thu, 01 Jan 1970")),
    ).toEqual([]);
    expect(writes.length).toBe(writesBeforeCapture);
    Reflect.deleteProperty(document, "cookie");

    act(() => answerCookiebot({ statistics: true }));
    posthog.capture("after_answer");

    expect(posthog.get_distinct_id()).toBe(WEBSITE_ID);
    expect(posthog.config.opt_out_persistence_by_default).toBe(true);
    expect(captured).toContain("after_answer");
    expect(sharedCookie()).toMatchObject({ $device_id: WEBSITE_ID });
  });

  it("deletes the cookie when the visitor declines after Cookiebot asks again", async () => {
    installCookiebot({ statistics: true });
    await renderPostHog();
    act(() => {
      if (window.Cookiebot) window.Cookiebot.hasResponse = false;
      window.dispatchEvent(new Event("CookiebotOnLoad"));
    });

    act(() => answerCookiebot({}));

    expect(sharedCookie()).toBeNull();
    expect(storedPostHogKeys()).toEqual([]);
  });

  it("waits for the session before taking the website's id", async () => {
    installCookiebot();
    const { posthog, setAnalyticsIdentitySignedIn } = await renderPostHog({
      sessionKnown: false,
    });
    const pageID = posthog.get_distinct_id();

    act(() => answerCookiebot({ statistics: true }));

    // Nothing is sent or stored under the page's id meanwhile.
    expect(posthog.get_distinct_id()).toBe(pageID);
    expect(posthog.has_opted_out_capturing()).toBe(true);
    expect(window.localStorage.getItem("agpt_anonymous_id")).toBeNull();
    expect(sharedCookie()).toEqual({
      distinct_id: WEBSITE_ID,
      $device_id: WEBSITE_ID,
    });
    expect(captured).toEqual([]);

    act(() => setAnalyticsIdentitySignedIn(false));

    expect(posthog.get_distinct_id()).toBe(WEBSITE_ID);
    expect(window.localStorage.getItem("agpt_anonymous_id")).toBe(WEBSITE_ID);
    expect(sharedCookie()).toMatchObject({ $device_id: WEBSITE_ID });
    // The landing page view was held, and goes out under the website's id.
    await vi.waitFor(() => expect(captured).toContain("$pageview"));
    expect(
      capturedEvents.find((event) => event.event === "$pageview")?.properties
        .distinct_id,
    ).toBe(WEBSITE_ID);
  });

  it("starts in memory on a stored grant until the session is known", async () => {
    installCookiebot({ statistics: true });
    window.localStorage.setItem("agpt_posthog_consent", "1");

    const { posthog, setAnalyticsIdentitySignedIn } = await renderPostHog({
      sessionKnown: false,
    });

    expect(posthog.config.persistence).toBe("memory");
    expect(posthog.has_opted_out_capturing()).toBe(true);
    expect(sharedCookie()).toEqual({
      distinct_id: WEBSITE_ID,
      $device_id: WEBSITE_ID,
    });

    act(() => setAnalyticsIdentitySignedIn(true));

    expect(posthog.config.persistence).toBe("localStorage+cookie");
    expect(posthog.get_distinct_id()).not.toBe(WEBSITE_ID);
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
