import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  installGtagShim,
  removeGtagShim,
} from "@/tests/integrations/gtag-shim";
import {
  answerCookiebot,
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";
import {
  CONVERSION_SEND_TIMEOUT_MS,
  getSubscriptionValue,
  parseConversionLabels,
  trackAdsConversion,
  trackAdsConversionBeforeNavigation,
  trackAdsPageView,
} from "./google-ads";

function answerBanner(marketing: boolean) {
  answerCookiebot({ statistics: true, marketing });
}

let pushed: unknown[][] = [];

describe("trackAdsConversion", () => {
  beforeEach(() => {
    pushed = installGtagShim();
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_ADS_ID", "AW-123");
    vi.stubEnv(
      "NEXT_PUBLIC_GOOGLE_ADS_CONVERSION_LABELS",
      "sign_up=SIGNUP,subscribe=SUB,trial_started=TS",
    );
    configureCookiebot();
    installCookiebot();
    answerBanner(true);
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    removeGtagShim();
    removeCookiebot();
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
    // Unanswered: Consent Mode denies ad_user_data for every visitor until
    // the banner is answered, so nothing identifying goes out.
    removeCookiebot();
    installCookiebot();

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

  it("withholds the identifiers when no banner is configured", () => {
    vi.stubEnv("NEXT_PUBLIC_COOKIEBOT_CBID", "");

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

  it("sends a trial start to its own action", () => {
    trackAdsConversion("trial_started", {
      transactionID: "user-1",
      email: "ada@example.com",
    });

    expect(pushed).toEqual([
      [
        "event",
        "conversion",
        {
          send_to: "AW-123/TS",
          transaction_id: "user-1",
          user_data: { email: "ada@example.com" },
        },
      ],
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

describe("trackAdsConversionBeforeNavigation", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    pushed = installGtagShim();
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_ADS_ID", "AW-123");
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_ADS_CONVERSION_LABELS", "begin_checkout=BC");
    configureCookiebot();
    installCookiebot();
    answerBanner(true);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllEnvs();
    removeGtagShim();
    removeCookiebot();
  });

  function navigateAfterConversion() {
    const navigate = vi.fn();
    void trackAdsConversionBeforeNavigation("begin_checkout", {
      value: 50,
      transactionID: "cs_123",
    }).then(navigate);
    return navigate;
  }

  function sendCallback() {
    const params = pushed[0][2] as { event_callback: () => void };
    params.event_callback();
  }

  it("sends the same conversion as trackAdsConversion, plus a callback", () => {
    navigateAfterConversion();

    expect(pushed).toEqual([
      [
        "event",
        "conversion",
        {
          send_to: "AW-123/BC",
          value: 50,
          currency: "USD",
          transaction_id: "cs_123",
          event_callback: expect.any(Function),
        },
      ],
    ]);
  });

  it("navigates once the tag reports the hit sent", async () => {
    const navigate = navigateAfterConversion();
    await vi.advanceTimersByTimeAsync(0);
    expect(navigate).not.toHaveBeenCalled();

    sendCallback();
    await vi.advanceTimersByTimeAsync(0);
    expect(navigate).toHaveBeenCalledOnce();

    await vi.advanceTimersByTimeAsync(CONVERSION_SEND_TIMEOUT_MS);
    expect(navigate).toHaveBeenCalledOnce();
  });

  it("navigates after the timeout when the tag never reports back", async () => {
    const navigate = navigateAfterConversion();

    await vi.advanceTimersByTimeAsync(CONVERSION_SEND_TIMEOUT_MS - 1);
    expect(navigate).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect(navigate).toHaveBeenCalledOnce();

    sendCallback();
    await vi.advanceTimersByTimeAsync(CONVERSION_SEND_TIMEOUT_MS);
    expect(navigate).toHaveBeenCalledOnce();
  });

  it("navigates straight away when the Ads tag is not configured", async () => {
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_ADS_ID", "");

    const navigate = navigateAfterConversion();
    await vi.advanceTimersByTimeAsync(0);

    expect(navigate).toHaveBeenCalledOnce();
    expect(pushed).toEqual([]);
  });

  it("navigates straight away for an action without a label", async () => {
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_ADS_CONVERSION_LABELS", "sign_up=SIGNUP");

    const navigate = navigateAfterConversion();
    await vi.advanceTimersByTimeAsync(0);

    expect(navigate).toHaveBeenCalledOnce();
    expect(pushed).toEqual([]);
  });

  it("navigates straight away when gtag is not on the page", async () => {
    removeGtagShim();

    const navigate = navigateAfterConversion();
    await vi.advanceTimersByTimeAsync(0);

    expect(navigate).toHaveBeenCalledOnce();
    expect(vi.getTimerCount()).toBe(0);
  });

  it("navigates straight away when the tag throws", async () => {
    window.gtag = () => {
      throw new Error("tag broke");
    };

    const navigate = navigateAfterConversion();
    await vi.advanceTimersByTimeAsync(0);

    expect(navigate).toHaveBeenCalledOnce();
    expect(vi.getTimerCount()).toBe(0);
    expect(trackAdsConversion("begin_checkout")).toBe(false);
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
        " sign_up=AbC , subscribe=DeF,unknown=X,,broken,top_up= ,trial_started=GhI",
      ),
    ).toEqual({ sign_up: "AbC", subscribe: "DeF", trial_started: "GhI" });
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
