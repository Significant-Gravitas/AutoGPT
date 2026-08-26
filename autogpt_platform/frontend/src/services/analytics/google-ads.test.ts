import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  installGtagShim,
  removeGtagShim,
} from "@/tests/integrations/gtag-shim";
import { consent } from "@/services/consent/cookies";
import {
  getSubscriptionValue,
  parseConversionLabels,
  trackAdsConversion,
  trackAdsPageView,
} from "./google-ads";

function answerBanner(advertising: boolean) {
  consent.save({
    hasConsented: true,
    timestamp: 1,
    analytics: true,
    monitoring: true,
    advertising,
  });
}

let pushed: unknown[][] = [];

describe("trackAdsConversion", () => {
  beforeEach(() => {
    pushed = installGtagShim();
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_ADS_ID", "AW-123");
    vi.stubEnv(
      "NEXT_PUBLIC_GOOGLE_ADS_CONVERSION_LABELS",
      "sign_up=SIGNUP,subscribe=SUB",
    );
    answerBanner(true);
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    removeGtagShim();
    consent.clear();
  });

  it("sends the conversion to the action label with value, dedup id and user data", () => {
    const sent = trackAdsConversion("subscribe", {
      value: 50,
      transactionID: "cs_123",
      email: "ada@example.com",
    });

    expect(sent).toBe(true);
    expect(pushed).toEqual([
      [
        "event",
        "conversion",
        {
          send_to: "AW-123/SUB",
          value: 50,
          currency: "USD",
          transaction_id: "cs_123",
          user_data: { email: "ada@example.com" },
        },
      ],
    ]);
  });

  it("withholds the identifiers until the banner is answered", () => {
    // Unanswered: Consent Mode denies ad_user_data in the EEA/UK/CH and the
    // browser can't tell which region it's in, so nothing identifying goes out.
    consent.clear();

    trackAdsConversion("subscribe", {
      value: 50,
      transactionID: "cs_123",
      email: "ada@example.com",
    });

    expect(pushed).toEqual([
      [
        "event",
        "conversion",
        { send_to: "AW-123/SUB", value: 50, currency: "USD" },
      ],
    ]);
  });

  it("withholds the identifiers when advertising was rejected", () => {
    answerBanner(false);

    trackAdsConversion("subscribe", {
      transactionID: "cs_123",
      email: "ada@example.com",
    });

    expect(pushed).toEqual([
      ["event", "conversion", { send_to: "AW-123/SUB" }],
    ]);
  });

  it("sends only the destination when no options are given", () => {
    trackAdsConversion("sign_up");

    expect(pushed).toEqual([
      ["event", "conversion", { send_to: "AW-123/SIGNUP" }],
    ]);
  });

  it("does nothing when the Google Ads tag is not configured", () => {
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_ADS_ID", "");

    expect(trackAdsConversion("sign_up")).toBe(false);
    expect(pushed).toEqual([]);
  });

  it("does nothing for an action without a label", () => {
    expect(trackAdsConversion("top_up")).toBe(false);
    expect(pushed).toEqual([]);
  });
});

describe("trackAdsPageView", () => {
  beforeEach(() => {
    pushed = installGtagShim();
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    removeGtagShim();
  });

  it("sends a page_view to the Ads tag only", () => {
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_ADS_ID", "AW-123");

    trackAdsPageView("/library");

    expect(pushed).toEqual([
      ["event", "page_view", { send_to: "AW-123", page_path: "/library" }],
    ]);
  });

  it("does nothing when the tag is not configured", () => {
    expect(trackAdsPageView("/library")).toBe(false);
    expect(pushed).toEqual([]);
  });
});

describe("parseConversionLabels", () => {
  it("reads key=label pairs and ignores unknown or malformed parts", () => {
    expect(
      parseConversionLabels(
        " sign_up=AbC , subscribe=DeF,unknown=X,,broken,top_up= ",
      ),
    ).toEqual({ sign_up: "AbC", subscribe: "DeF" });
  });

  it("returns nothing for an unset value", () => {
    expect(parseConversionLabels("")).toEqual({});
  });
});

describe("getSubscriptionValue", () => {
  it("prices monthly plans at the monthly rate", () => {
    expect(getSubscriptionValue("PRO", "monthly")).toBe(50);
    expect(getSubscriptionValue("MAX", "monthly")).toBe(320);
  });

  it("prices yearly plans at the discounted annual total", () => {
    expect(getSubscriptionValue("PRO", "yearly")).toBe(510);
    expect(getSubscriptionValue("MAX", "yearly")).toBe(3264);
  });

  it("has no value for unknown plans", () => {
    expect(getSubscriptionValue("TEAM", "monthly")).toBeUndefined();
    expect(getSubscriptionValue(null, null)).toBeUndefined();
  });
});
