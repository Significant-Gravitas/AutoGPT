import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";
import { getDatafastAttribution } from "./datafast-attribution";

function setCookie(value: string) {
  vi.spyOn(document, "cookie", "get").mockReturnValue(value);
}

beforeEach(() => {
  configureCookiebot();
  installCookiebot({ statistics: true });
});

afterEach(() => {
  vi.restoreAllMocks();
  removeCookiebot();
  vi.unstubAllEnvs();
});

describe("getDatafastAttribution", () => {
  it("returns both header values when both cookies are present", () => {
    setCookie("datafast_visitor_id=vis_1; datafast_session_id=ses_1");
    expect(getDatafastAttribution()).toEqual({
      "X-Datafast-Visitor-Id": "vis_1",
      "X-Datafast-Session-Id": "ses_1",
    });
  });

  it("includes only the cookies that are present", () => {
    setCookie("datafast_visitor_id=vis_1");
    expect(getDatafastAttribution()).toEqual({
      "X-Datafast-Visitor-Id": "vis_1",
    });
  });

  it("returns an empty object when no DataFast cookies exist", () => {
    setCookie("other=1");
    expect(getDatafastAttribution()).toEqual({});
  });

  it("does not throw on a malformed percent-encoded unrelated cookie", () => {
    setCookie("other=%E0%A4%A; datafast_visitor_id=vis_1");
    expect(() => getDatafastAttribution()).not.toThrow();
    expect(getDatafastAttribution()).toEqual({
      "X-Datafast-Visitor-Id": "vis_1",
    });
  });

  it("skips a DataFast cookie whose own value is malformed", () => {
    setCookie("datafast_visitor_id=%E0%A4%A; datafast_session_id=ses_1");
    expect(() => getDatafastAttribution()).not.toThrow();
    expect(getDatafastAttribution()).toEqual({
      "X-Datafast-Session-Id": "ses_1",
    });
  });
});

describe("getDatafastAttribution without analytics consent", () => {
  beforeEach(() => {
    setCookie("datafast_visitor_id=vis_1; datafast_session_id=ses_1");
  });

  it("sends nothing once the visitor declines analytics", () => {
    installCookiebot({ marketing: true });
    expect(getDatafastAttribution()).toEqual({});
  });

  it("sends nothing while the answer is pending", () => {
    installCookiebot();
    expect(getDatafastAttribution()).toEqual({});
  });

  it("sends nothing without a consent manager", () => {
    vi.stubEnv("NEXT_PUBLIC_COOKIEBOT_CBID", "");
    expect(getDatafastAttribution()).toEqual({});
  });
});
