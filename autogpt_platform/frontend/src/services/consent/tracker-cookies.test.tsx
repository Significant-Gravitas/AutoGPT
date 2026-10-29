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
const UNRELATED_COOKIES = [
  "__Secure-better-auth.session_token",
  "admin-impersonate-user-id",
  "ph_phc_test_posthog",
  "_gallery",
];

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

function storeAnswer(answer: { statistics: boolean; marketing: boolean }) {
  const value = `{stamp:'abc',necessary:true,preferences:false,statistics:${answer.statistics},marketing:${answer.marketing},method:'explicit',ver:1,utc:1,region:'de'}`;
  document.cookie = `CookieConsent=${encodeURIComponent(value)}; Path=/`;
}

// What an earlier accept left behind: the Google tag and DataFast write on
// .agpt.co, and a host-only copy can sit beside it.
function seedEarlierAccept() {
  [...ANALYTICS_COOKIES, ...ADVERTISING_COOKIES].forEach((name) => {
    document.cookie = `${name}=1; Path=/; Domain=.agpt.co; Secure`;
  });
  document.cookie = "_ga=host; Path=/; Secure";
  document.cookie =
    "__Secure-better-auth.session_token=session; Path=/; Secure";
  document.cookie = "admin-impersonate-user-id=user-2; Path=/; Secure";
  document.cookie = "ph_phc_test_posthog=x; Path=/; Domain=.agpt.co; Secure";
  document.cookie = "_gallery=1; Path=/; Secure";
}

beforeEach(() => {
  setPageURL("https://platform.agpt.co/");
  configureCookiebot();
  seedEarlierAccept();
});

afterEach(() => {
  removeCookiebot();
  new Set(cookieNames()).forEach((name) => {
    const expired = `${name}=; Path=/; Expires=Thu, 01 Jan 1970 00:00:00 GMT`;
    document.cookie = expired;
    document.cookie = `${expired}; Domain=.agpt.co`;
  });
  setPageURL("http://localhost:3000/");
  vi.unstubAllEnvs();
});

describe("tracker cookies after a declining answer", () => {
  it("deletes the analytics and advertising cookies when the visitor declines", () => {
    installCookiebot();
    render(<TrackerCookieCleanup />);

    act(() => answerCookiebot({}));

    expect(cookieNames()).toEqual(expect.arrayContaining(UNRELATED_COOKIES));
    [...ANALYTICS_COOKIES, ...ADVERTISING_COOKIES].forEach((name) =>
      expect(cookieNames()).not.toContain(name),
    );
  });

  it("deletes only the analytics cookies when the visitor allows marketing", () => {
    installCookiebot();
    render(<TrackerCookieCleanup />);

    act(() => answerCookiebot({ marketing: true }));

    ANALYTICS_COOKIES.forEach((name) =>
      expect(cookieNames()).not.toContain(name),
    );
    expect(cookieNames()).toEqual(
      expect.arrayContaining([...ADVERTISING_COOKIES, ...UNRELATED_COOKIES]),
    );
  });

  it("deletes only the advertising cookies when the visitor allows statistics", () => {
    installCookiebot();
    render(<TrackerCookieCleanup />);

    act(() => answerCookiebot({ statistics: true }));

    ADVERTISING_COOKIES.forEach((name) =>
      expect(cookieNames()).not.toContain(name),
    );
    expect(cookieNames()).toEqual(
      expect.arrayContaining([...ANALYTICS_COOKIES, ...UNRELATED_COOKIES]),
    );
  });

  it("deletes them when the visitor withdraws an earlier grant", () => {
    installCookiebot({ statistics: true, marketing: true });
    render(<TrackerCookieCleanup />);
    expect(cookieNames()).toEqual(
      expect.arrayContaining([...ANALYTICS_COOKIES, ...ADVERTISING_COOKIES]),
    );

    act(() => answerCookiebot({}));

    [...ANALYTICS_COOKIES, ...ADVERTISING_COOKIES].forEach((name) =>
      expect(cookieNames()).not.toContain(name),
    );
    expect(cookieNames()).toEqual(expect.arrayContaining(UNRELATED_COOKIES));
  });

  it("deletes them on page load when the stored answer already declined", () => {
    storeAnswer({ statistics: false, marketing: false });

    render(<TrackerCookieCleanup />);

    [...ANALYTICS_COOKIES, ...ADVERTISING_COOKIES].forEach((name) =>
      expect(cookieNames()).not.toContain(name),
    );
    expect(cookieNames()).toEqual(
      expect.arrayContaining([...UNRELATED_COOKIES, "CookieConsent"]),
    );
  });
});

describe("tracker cookies without a declining answer", () => {
  it("keeps them while the visitor allows both categories", () => {
    installCookiebot({ statistics: true, marketing: true });
    render(<TrackerCookieCleanup />);

    act(() => answerCookiebot({ statistics: true, marketing: true }));

    expect(cookieNames()).toEqual(
      expect.arrayContaining([
        ...ANALYTICS_COOKIES,
        ...ADVERTISING_COOKIES,
        ...UNRELATED_COOKIES,
      ]),
    );
  });

  it("keeps them while the answer is pending", () => {
    render(<TrackerCookieCleanup />);
    act(() => {
      installCookiebot();
      window.dispatchEvent(new Event("CookiebotOnDialogDisplay"));
    });

    expect(cookieNames()).toEqual(
      expect.arrayContaining([...ANALYTICS_COOKIES, ...ADVERTISING_COOKIES]),
    );
  });

  it("keeps them while Cookiebot asks again over a stored decline", () => {
    storeAnswer({ statistics: false, marketing: false });
    installCookiebot();

    render(<TrackerCookieCleanup />);
    act(() => {
      window.dispatchEvent(new Event("CookiebotOnLoad"));
    });

    expect(cookieNames()).toEqual(
      expect.arrayContaining([...ANALYTICS_COOKIES, ...ADVERTISING_COOKIES]),
    );
  });

  it("keeps them without a consent manager", () => {
    vi.stubEnv("NEXT_PUBLIC_COOKIEBOT_CBID", "");
    storeAnswer({ statistics: false, marketing: false });

    render(<TrackerCookieCleanup />);

    expect(cookieNames()).toEqual(
      expect.arrayContaining([...ANALYTICS_COOKIES, ...ADVERTISING_COOKIES]),
    );
  });
});
