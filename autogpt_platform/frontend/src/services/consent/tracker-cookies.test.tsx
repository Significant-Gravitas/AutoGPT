import {
  answerCookiebot,
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";
import { act, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TrackerCookieCleanup } from "./TrackerCookieCleanup";

const ANALYTICS_COOKIES = [
  "_ga",
  "_ga_ABC123",
  "datafast_visitor_id",
  "datafast_session_id",
];
const ADVERTISING_COOKIES = ["_gcl_au", "_gcl_aw"];
const TRACKER_COOKIES = [...ANALYTICS_COOKIES, ...ADVERTISING_COOKIES];
const UNRELATED_COOKIES = [
  "__Secure-better-auth.session_token",
  "admin-impersonate-user-id",
  "ph_phc_test_posthog",
  "_gallery",
];

const ANALYTICS_KEYS = ["datafast_pageview_state"];
const ADVERTISING_KEYS = ["_gcl_ls", "_gcl_dc"];
const UNRELATED_KEYS = ["agpt_anonymous_id", "ph_phc_test_posthog"];

interface Answer {
  statistics?: boolean;
  marketing?: boolean;
}

function setPageURL(url: string) {
  const { happyDOM } = window as unknown as {
    happyDOM?: { setURL: (url: string) => void };
  };
  happyDOM?.setURL(url);
}

function cookieNames() {
  return document.cookie
    .split(";")
    .map((part) => part.trim().split("=")[0])
    .filter(Boolean);
}

function storageKeys() {
  return [...Object.keys(localStorage), ...Object.keys(sessionStorage)];
}

// Cookiebot's stored answer; `id` stands in for its consent ID and time,
// which change whenever the visitor answers.
function storeAnswer(answer: Answer, id: string) {
  const value = `{stamp:'${id}',necessary:true,preferences:false,statistics:${Boolean(answer.statistics)},marketing:${Boolean(answer.marketing)},method:'explicit',ver:1,utc:1700000000000,region:'de'}`;
  document.cookie = `CookieConsent=${encodeURIComponent(value)}; Path=/`;
}

// The visitor answers the banner on this page.
function answerOnPage(answer: Answer, id: string) {
  act(() => {
    storeAnswer(answer, id);
    answerCookiebot(answer);
  });
}

// What the trackers stored while allowed: the Google tag and DataFast write
// their cookies on .agpt.co.
function seedTrackers() {
  TRACKER_COOKIES.forEach((name) => {
    document.cookie = `${name}=1; Path=/; Domain=.agpt.co; Secure`;
  });
  [...ANALYTICS_KEYS, ...ADVERTISING_KEYS].forEach((key) =>
    localStorage.setItem(key, "1"),
  );
  sessionStorage.setItem("datafast_session_state", "1");
}

function seedUnrelated() {
  document.cookie =
    "__Secure-better-auth.session_token=session; Path=/; Secure";
  document.cookie = "admin-impersonate-user-id=user-2; Path=/; Secure";
  document.cookie = "ph_phc_test_posthog=x; Path=/; Domain=.agpt.co; Secure";
  document.cookie = "_gallery=1; Path=/; Secure";
  UNRELATED_KEYS.forEach((key) => localStorage.setItem(key, "1"));
}

function expectTrackersKept(cookies: string[], keys: string[]) {
  expect(cookieNames()).toEqual(expect.arrayContaining(cookies));
  expect(storageKeys()).toEqual(expect.arrayContaining(keys));
}

function expectTrackersGone(cookies: string[], keys: string[]) {
  cookies.forEach((name) => expect(cookieNames()).not.toContain(name));
  keys.forEach((key) => expect(storageKeys()).not.toContain(key));
}

function expectAllTrackersGone() {
  expectTrackersGone(TRACKER_COOKIES, [
    ...ANALYTICS_KEYS,
    ...ADVERTISING_KEYS,
    "datafast_session_state",
  ]);
}

function expectAllTrackersKept() {
  expectTrackersKept(TRACKER_COOKIES, [...ANALYTICS_KEYS, ...ADVERTISING_KEYS]);
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

beforeEach(() => {
  setPageURL("https://platform.agpt.co/");
  configureCookiebot();
  seedTrackers();
  seedUnrelated();
});

afterEach(() => {
  Reflect.deleteProperty(document, "cookie");
  removeCookiebot();
  new Set(cookieNames()).forEach((name) => {
    const expired = `${name}=; Path=/; Expires=Thu, 01 Jan 1970 00:00:00 GMT`;
    document.cookie = expired;
    document.cookie = `${expired}; Domain=.agpt.co`;
  });
  localStorage.clear();
  sessionStorage.clear();
  setPageURL("http://localhost:3000/");
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
});

describe("trackers after an answer given on the page", () => {
  it("deletes both categories' cookies and storage when the visitor declines", () => {
    installCookiebot();
    render(<TrackerCookieCleanup />);

    answerOnPage({}, "decline-1");

    expectAllTrackersGone();
    expect(cookieNames()).toEqual(expect.arrayContaining(UNRELATED_COOKIES));
    expect(storageKeys()).toEqual(expect.arrayContaining(UNRELATED_KEYS));
  });

  it("deletes only the analytics trackers when the visitor allows marketing", () => {
    installCookiebot();
    render(<TrackerCookieCleanup />);

    answerOnPage({ marketing: true }, "decline-1");

    expectTrackersGone(ANALYTICS_COOKIES, ANALYTICS_KEYS);
    expectTrackersKept(ADVERTISING_COOKIES, ADVERTISING_KEYS);
  });

  it("deletes only the advertising trackers when the visitor allows statistics", () => {
    installCookiebot();
    render(<TrackerCookieCleanup />);

    answerOnPage({ statistics: true }, "decline-1");

    expectTrackersGone(ADVERTISING_COOKIES, ADVERTISING_KEYS);
    expectTrackersKept(ANALYTICS_COOKIES, ANALYTICS_KEYS);
  });

  it("deletes them when the visitor withdraws an earlier grant", () => {
    storeAnswer({ statistics: true, marketing: true }, "grant-1");
    installCookiebot({ statistics: true, marketing: true });
    render(<TrackerCookieCleanup />);
    expectAllTrackersKept();

    answerOnPage({}, "withdraw-1");

    expectAllTrackersGone();
  });

  it("expires each cookie on the host and on every parent domain", () => {
    installCookiebot();
    render(<TrackerCookieCleanup />);
    const writes = recordCookieWrites();

    answerOnPage({}, "decline-1");

    const deletions = writes.filter(
      (value) =>
        value.startsWith("_ga=;") &&
        value.includes("Expires=Thu, 01 Jan 1970 00:00:00 GMT"),
    );
    expect(deletions).toEqual(
      expect.arrayContaining([
        expect.not.stringContaining("Domain="),
        expect.stringContaining("Domain=.platform.agpt.co"),
        expect.stringContaining("Domain=.agpt.co"),
      ]),
    );
  });

  it("cleans even when the same answer was cleaned on an earlier page", () => {
    installCookiebot();
    const first = render(<TrackerCookieCleanup />);
    answerOnPage({}, "decline-1");
    first.unmount();
    removeCookiebot();
    seedTrackers();

    installCookiebot();
    render(<TrackerCookieCleanup />);
    answerOnPage({}, "decline-1");

    expectAllTrackersGone();
  });
});

describe("trackers after an answer found on page load", () => {
  it("deletes them when the stored answer already declined", () => {
    storeAnswer({}, "stored-1");

    render(<TrackerCookieCleanup />);

    expectAllTrackersGone();
    expect(cookieNames()).toEqual(
      expect.arrayContaining([...UNRELATED_COOKIES, "CookieConsent"]),
    );
  });

  it("deletes them once per answer, not on every page load", () => {
    storeAnswer({}, "stored-1");
    installCookiebot({});
    render(<TrackerCookieCleanup />).unmount();
    expectAllTrackersGone();
    // agpt.co, where the visitor allowed them, writes them again.
    seedTrackers();

    render(<TrackerCookieCleanup />);
    act(() => {
      window.dispatchEvent(new Event("CookiebotOnConsentReady"));
      window.dispatchEvent(new Event("CookiebotOnDecline"));
    });

    expectAllTrackersKept();
  });

  it("deletes them again for a new answer", () => {
    storeAnswer({}, "stored-1");
    render(<TrackerCookieCleanup />).unmount();
    seedTrackers();
    storeAnswer({}, "stored-2");

    render(<TrackerCookieCleanup />);

    expectAllTrackersGone();
  });

  it("waits for uc.js to confirm a stored decline", () => {
    vi.spyOn(document, "readyState", "get").mockReturnValue("interactive");
    storeAnswer({}, "stored-1");
    render(<TrackerCookieCleanup />);
    expectAllTrackersKept();

    act(() => {
      installCookiebot({});
      window.dispatchEvent(new Event("CookiebotOnConsentReady"));
    });

    expectAllTrackersGone();
  });
});

describe("trackers without a declining answer", () => {
  it("keeps them while the visitor allows both categories", () => {
    storeAnswer({ statistics: true, marketing: true }, "grant-1");
    installCookiebot({ statistics: true, marketing: true });
    render(<TrackerCookieCleanup />);

    answerOnPage({ statistics: true, marketing: true }, "grant-2");

    expectAllTrackersKept();
  });

  it("keeps them while the answer is pending", () => {
    render(<TrackerCookieCleanup />);
    act(() => {
      installCookiebot();
      window.dispatchEvent(new Event("CookiebotOnDialogDisplay"));
    });

    expectAllTrackersKept();
  });

  it("keeps them while Cookiebot asks again over a stored decline", () => {
    storeAnswer({}, "stored-1");
    installCookiebot();

    render(<TrackerCookieCleanup />);
    act(() => {
      window.dispatchEvent(new Event("CookiebotOnLoad"));
    });

    expectAllTrackersKept();
  });

  it("keeps them without a consent manager", () => {
    vi.stubEnv("NEXT_PUBLIC_COOKIEBOT_CBID", "");
    storeAnswer({}, "stored-1");

    render(<TrackerCookieCleanup />);

    expectAllTrackersKept();
  });
});
