import {
  answerCookiebot,
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  captureFirstLanding,
  followAnalyticsConsentForIdentity,
  getAnonymousID,
  getPostHogDeviceID,
  readFirstLanding,
  resetAnonymousID,
  resetAnonymousIDForTests,
} from "../anonymous-id";

function setAnalyticsConsent(analytics: boolean): void {
  configureCookiebot();
  installCookiebot({ statistics: analytics });
}

function landOn(path: string, referrer = ""): void {
  window.history.pushState({}, "", path);
  Object.defineProperty(document, "referrer", {
    value: referrer,
    configurable: true,
  });
}

const ANONYMOUS_ID_KEY = "agpt_anonymous_id";
const FIRST_LANDING_KEY = "agpt_first_landing";

beforeEach(() => {
  window.localStorage.clear();
  resetAnonymousIDForTests();
  setAnalyticsConsent(true);
  landOn("/");
  vi.stubEnv("NEXT_PUBLIC_POSTHOG_KEY", "phc_test");
});

afterEach(() => {
  removeCookiebot();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("getAnonymousID", () => {
  it("mints one id, persists it, and returns the same id afterwards", () => {
    const first = getAnonymousID();

    expect(first).toBeTruthy();
    expect(window.localStorage.getItem(ANONYMOUS_ID_KEY)).toBe(first);
    expect(getAnonymousID()).toBe(first);

    resetAnonymousIDForTests();
    expect(getAnonymousID()).toBe(first);
  });

  it("adopts an existing PostHog device id so returning visitors keep history", () => {
    window.localStorage.setItem(
      "ph_phc_test_posthog",
      JSON.stringify({ $device_id: "device-123" }),
    );

    expect(getAnonymousID()).toBe("device-123");
    expect(getPostHogDeviceID()).toBe("device-123");
  });

  it("ignores unreadable PostHog persistence", () => {
    window.localStorage.setItem("ph_phc_test_posthog", "not json");

    expect(getPostHogDeviceID()).toBeNull();
    expect(getAnonymousID()).not.toBe("not json");
  });

  it("keeps an in-memory id when storage is blocked", () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });

    const id = getAnonymousID();

    expect(id).toBeTruthy();
    expect(getAnonymousID()).toBe(id);
  });
});

describe("resetAnonymousID", () => {
  it("forgets the id and first landing so the next visitor starts fresh", () => {
    const first = getAnonymousID();
    captureFirstLanding();
    expect(readFirstLanding()).not.toBeNull();

    resetAnonymousID();

    expect(window.localStorage.getItem(ANONYMOUS_ID_KEY)).toBe(
      getAnonymousID(),
    );
    expect(readFirstLanding()).toBeNull();
    expect(getAnonymousID()).not.toBe(first);
  });

  it("keeps the replacement identity after the in-memory cache is cleared", () => {
    window.localStorage.setItem(
      "ph_phc_test_posthog",
      JSON.stringify({ $device_id: "old-device" }),
    );
    expect(getAnonymousID()).toBe("old-device");

    resetAnonymousID("fresh-device");
    resetAnonymousIDForTests();

    expect(getAnonymousID()).toBe("fresh-device");
  });
});

describe("first landing", () => {
  it("records the first page once and reads it back", () => {
    window.history.replaceState(
      null,
      "",
      "/pricing?utm_source=x&utm_medium=cpc&utm_campaign=launch",
    );

    captureFirstLanding();
    window.history.replaceState(null, "", "/somewhere-else");
    captureFirstLanding();

    const landing = readFirstLanding();
    expect(landing?.path).toBe(
      "/pricing?utm_source=x&utm_medium=cpc&utm_campaign=launch",
    );
    expect(landing?.utm_source).toBe("x");
    expect(landing?.utm_medium).toBe("cpc");
    expect(landing?.utm_campaign).toBe("launch");
    expect(landing?.referrer).toBeNull();
    expect(landing?.at).toBeTruthy();
  });

  it("returns null when nothing was captured or the record is corrupt", () => {
    expect(readFirstLanding()).toBeNull();

    window.localStorage.setItem(FIRST_LANDING_KEY, "{");
    expect(readFirstLanding()).toBeNull();
  });
});

describe("first landing redaction", () => {
  it("records the route but never a one-time token from the path", () => {
    landOn("/link/super-secret-invite-token");

    captureFirstLanding();

    const landing = readFirstLanding();
    expect(landing?.path).toBe("/link");
    expect(JSON.stringify(landing)).not.toContain("super-secret-invite-token");
  });

  it("drops sensitive query params while keeping campaign tags", () => {
    landOn("/marketplace?utm_source=newsletter&code=oauth-code&token=secret");

    captureFirstLanding();

    const landing = readFirstLanding();
    expect(landing?.path).toBe("/marketplace?utm_source=newsletter");
    expect(landing?.utm_source).toBe("newsletter");
    expect(JSON.stringify(landing)).not.toContain("oauth-code");
    expect(JSON.stringify(landing)).not.toContain("secret");
  });

  it("redacts a same-origin referrer that carries a token", () => {
    landOn(
      "/marketplace",
      `${window.location.origin}/share/secret-share-token`,
    );

    captureFirstLanding();

    const landing = readFirstLanding();
    expect(landing?.referrer).toBe(`${window.location.origin}/share`);
  });

  it("keeps an external referrer's origin and path, without its query", () => {
    landOn("/marketplace", "https://news.example.com/post?session=abc");

    captureFirstLanding();

    expect(readFirstLanding()?.referrer).toBe("https://news.example.com/post");
  });
});

describe("without analytics consent", () => {
  beforeEach(() => {
    setAnalyticsConsent(false);
  });

  it("mints a fresh id per page load and never stores it", () => {
    const first = getAnonymousID();

    expect(first).toBeTruthy();
    expect(getAnonymousID()).toBe(first);
    expect(window.localStorage.getItem(ANONYMOUS_ID_KEY)).toBeNull();

    resetAnonymousIDForTests();
    expect(getAnonymousID()).not.toBe(first);
  });

  it("neither reads a stored id nor adopts a leftover PostHog device id", () => {
    const getItem = vi.spyOn(Storage.prototype, "getItem");
    window.localStorage.setItem(ANONYMOUS_ID_KEY, "stored-id");
    window.localStorage.setItem(
      "ph_phc_test_posthog",
      JSON.stringify({ $device_id: "device-123" }),
    );
    document.cookie = `ph_phc_test_posthog=${encodeURIComponent(
      JSON.stringify({ $device_id: "cookie-device" }),
    )}; Path=/`;

    const id = getAnonymousID();

    expect(["stored-id", "device-123", "cookie-device"]).not.toContain(id);
    expect(getPostHogDeviceID()).toBeNull();
    expect(getItem).not.toHaveBeenCalled();
    document.cookie = "ph_phc_test_posthog=; Path=/; Max-Age=0";
  });

  it("keeps the landing in memory instead of storing it", () => {
    window.localStorage.setItem(
      FIRST_LANDING_KEY,
      JSON.stringify({ path: "/stored" }),
    );
    landOn("/pricing?utm_source=newsletter");

    captureFirstLanding();

    expect(window.localStorage.getItem(FIRST_LANDING_KEY)).toBe(
      JSON.stringify({ path: "/stored" }),
    );
    expect(readFirstLanding()?.path).toBe("/pricing?utm_source=newsletter");
    expect(readFirstLanding()?.utm_source).toBe("newsletter");
  });

  it("deletes what a visit with consent stored", () => {
    window.localStorage.setItem(ANONYMOUS_ID_KEY, "stored-id");
    window.localStorage.setItem(FIRST_LANDING_KEY, "{}");

    followAnalyticsConsentForIdentity()();

    expect(window.localStorage.getItem(ANONYMOUS_ID_KEY)).toBeNull();
    expect(window.localStorage.getItem(FIRST_LANDING_KEY)).toBeNull();
  });
});

describe("following analytics consent", () => {
  it("persists this page's id and landing on a grant, without a reload", () => {
    configureCookiebot();
    installCookiebot();
    landOn("/pricing?utm_source=newsletter");
    const id = getAnonymousID();
    captureFirstLanding();
    const unfollow = followAnalyticsConsentForIdentity();
    expect(window.localStorage.getItem(ANONYMOUS_ID_KEY)).toBeNull();

    answerCookiebot({ statistics: true });

    expect(window.localStorage.getItem(ANONYMOUS_ID_KEY)).toBe(id);
    expect(getAnonymousID()).toBe(id);
    expect(
      JSON.parse(window.localStorage.getItem(FIRST_LANDING_KEY) ?? "{}").path,
    ).toBe("/pricing?utm_source=newsletter");
    unfollow();
  });

  it("deletes the stored id and landing when consent is withdrawn", () => {
    const id = getAnonymousID();
    captureFirstLanding();
    const unfollow = followAnalyticsConsentForIdentity();
    expect(window.localStorage.getItem(ANONYMOUS_ID_KEY)).toBe(id);
    expect(window.localStorage.getItem(FIRST_LANDING_KEY)).not.toBeNull();

    answerCookiebot({});

    expect(window.localStorage.getItem(ANONYMOUS_ID_KEY)).toBeNull();
    expect(window.localStorage.getItem(FIRST_LANDING_KEY)).toBeNull();
    unfollow();
  });

  it("keeps a first landing stored on an earlier visit", () => {
    window.localStorage.setItem(
      FIRST_LANDING_KEY,
      JSON.stringify({ path: "/earlier" }),
    );
    landOn("/later");

    captureFirstLanding();
    followAnalyticsConsentForIdentity()();

    expect(readFirstLanding()?.path).toBe("/earlier");
  });
});
