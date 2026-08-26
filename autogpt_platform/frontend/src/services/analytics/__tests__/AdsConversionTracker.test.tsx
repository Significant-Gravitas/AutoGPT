import { render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  installGtagShim,
  removeGtagShim,
} from "@/tests/integrations/gtag-shim";
import { ACCOUNT_CREATED_COOKIE } from "../account-created-cookie";
import { consent } from "@/services/consent/cookies";

let pathname = "/copilot";
vi.mock("next/navigation", () => ({
  usePathname: () => pathname,
}));

const auth = vi.hoisted(() => ({
  state: {
    user: { id: "user-1", email: "ada@example.com" } as {
      id: string;
      email: string;
    } | null,
    isUserLoading: false,
  },
}));
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => auth.state,
}));

import { AdsConversionTracker } from "../AdsConversionTracker";

let pushed: unknown[][] = [];

function conversions() {
  return pushed.filter(([, name]) => name === "conversion");
}

function pageViews() {
  return pushed.filter(([, name]) => name === "page_view");
}

describe("AdsConversionTracker", () => {
  beforeEach(() => {
    pushed = installGtagShim();
    document.cookie = `${ACCOUNT_CREATED_COOKIE}=; Path=/; Max-Age=0`;
    window.history.replaceState({}, "", "/copilot");
    pathname = "/copilot";
    auth.state = {
      user: { id: "user-1", email: "ada@example.com" },
      isUserLoading: false,
    };
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_ADS_ID", "AW-123");
    vi.stubEnv(
      "NEXT_PUBLIC_GOOGLE_ADS_CONVERSION_LABELS",
      "sign_up=SU,subscribe=SB,top_up=TU",
    );
    // Identifiers only ride along on an affirmative yes, so the cases below
    // that assert them start from an answered banner.
    consent.save({
      hasConsented: true,
      timestamp: 1,
      analytics: true,
      monitoring: true,
      advertising: true,
    });
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    removeGtagShim();
    consent.clear();
  });

  it("fires sign_up once for a just-created account and clears the flag", () => {
    document.cookie = `${ACCOUNT_CREATED_COOKIE}=email; Path=/`;

    const { rerender } = render(<AdsConversionTracker />);
    rerender(<AdsConversionTracker />);

    expect(conversions()).toEqual([
      [
        "event",
        "conversion",
        {
          send_to: "AW-123/SU",
          transaction_id: "user-1",
          user_data: { email: "ada@example.com" },
        },
      ],
    ]);
    expect(document.cookie).not.toContain(`${ACCOUNT_CREATED_COOKIE}=email`);
  });

  it("waits for the session before firing sign_up", () => {
    document.cookie = `${ACCOUNT_CREATED_COOKIE}=google; Path=/`;
    auth.state = { user: null, isUserLoading: true };

    const { rerender } = render(<AdsConversionTracker />);
    expect(conversions()).toEqual([]);

    auth.state = {
      user: { id: "user-2", email: "bob@example.com" },
      isUserLoading: false,
    };
    rerender(<AdsConversionTracker />);

    expect(conversions()).toHaveLength(1);
    expect(conversions()[0][2]).toMatchObject({ transaction_id: "user-2" });
  });

  it("does not fire sign_up for a returning user", () => {
    render(<AdsConversionTracker />);

    expect(conversions()).toEqual([]);
  });

  it("picks up a flag set after mount on the next client-side navigation", () => {
    const { rerender } = render(<AdsConversionTracker />);
    expect(conversions()).toEqual([]);

    // The email signup action sets the flag, then router.replace()s onward.
    document.cookie = `${ACCOUNT_CREATED_COOKIE}=email; Path=/`;
    pathname = "/onboarding";
    rerender(<AdsConversionTracker />);

    expect(conversions()).toHaveLength(1);
    expect(conversions()[0][2]).toMatchObject({ send_to: "AW-123/SU" });
  });

  it("fires subscribe with the plan price when returning from Stripe", () => {
    window.history.replaceState(
      {},
      "",
      "/onboarding?step=2&subscription=success&session_id=cs_1&plan=MAX&cycle=yearly",
    );

    render(<AdsConversionTracker />);

    expect(conversions()).toEqual([
      [
        "event",
        "conversion",
        {
          send_to: "AW-123/SB",
          value: 3264,
          currency: "USD",
          transaction_id: "cs_1",
          user_data: { email: "ada@example.com" },
        },
      ],
    ]);
  });

  it("does not fire subscribe on a cancelled checkout", () => {
    window.history.replaceState(
      {},
      "",
      "/onboarding?step=1&subscription=cancelled",
    );

    render(<AdsConversionTracker />);

    expect(conversions()).toEqual([]);
  });

  it("fires top_up with the Stripe session as the dedup key", () => {
    window.history.replaceState(
      {},
      "",
      "/settings/billing?topup=success&session_id=cs_top_1",
    );

    render(<AdsConversionTracker />);

    expect(conversions()).toEqual([
      [
        "event",
        "conversion",
        {
          send_to: "AW-123/TU",
          transaction_id: "cs_top_1",
          user_data: { email: "ada@example.com" },
        },
      ],
    ]);
  });

  it("reports nothing while the session is absent, even once loading is done", () => {
    // The regression: on the post-signup client navigation the effect runs
    // with isUserLoading already false but no user yet. Reporting there sends
    // a conversion with no dedup id and no enhanced-conversion email.
    document.cookie = `${ACCOUNT_CREATED_COOKIE}=email; Path=/`;
    auth.state = { user: null, isUserLoading: false };

    const { rerender } = render(<AdsConversionTracker />);
    expect(conversions()).toEqual([]);
    expect(document.cookie).toContain(`${ACCOUNT_CREATED_COOKIE}=email`);

    auth.state = {
      user: { id: "user-3", email: "cleo@example.com" },
      isUserLoading: false,
    };
    rerender(<AdsConversionTracker />);

    expect(conversions()).toHaveLength(1);
    expect(conversions()[0][2]).toMatchObject({
      transaction_id: "user-3",
      user_data: { email: "cleo@example.com" },
    });
  });

  it("keeps the account-created flag when the tag has not loaded yet", () => {
    removeGtagShim();
    document.cookie = `${ACCOUNT_CREATED_COOKIE}=email; Path=/`;

    const { rerender } = render(<AdsConversionTracker />);
    expect(document.cookie).toContain(`${ACCOUNT_CREATED_COOKIE}=email`);

    // The tag loads afterInteractive; the next navigation retries.
    pushed = installGtagShim();
    pathname = "/library";
    rerender(<AdsConversionTracker />);

    expect(conversions()).toHaveLength(1);
    expect(document.cookie).not.toContain(`${ACCOUNT_CREATED_COOKIE}=email`);
  });

  it("retries the checkout return until it reaches the tag", () => {
    removeGtagShim();
    window.history.replaceState(
      {},
      "",
      "/settings/billing?subscription=success&session_id=cs_2&plan=PRO&cycle=monthly",
    );

    const { rerender } = render(<AdsConversionTracker />);

    pushed = installGtagShim();
    // The navigation that triggers the retry also strips the query — the
    // snapshot taken at mount is what has to carry it.
    window.history.replaceState({}, "", "/library");
    pathname = "/library";
    rerender(<AdsConversionTracker />);

    expect(conversions()).toHaveLength(1);
    expect(conversions()[0][2]).toMatchObject({
      send_to: "AW-123/SB",
      transaction_id: "cs_2",
    });
  });

  it("drops the identifiers when the visitor rejected advertising", () => {
    consent.save({
      hasConsented: true,
      timestamp: 1,
      analytics: true,
      monitoring: true,
      advertising: false,
    });
    document.cookie = `${ACCOUNT_CREATED_COOKIE}=email; Path=/`;

    render(<AdsConversionTracker />);

    // The aggregate conversion still counts — Consent Mode keeps it
    // cookieless — but nothing identifying rides along.
    expect(conversions()).toEqual([
      ["event", "conversion", { send_to: "AW-123/SU" }],
    ]);
  });

  it("keeps the identifiers when the visitor accepted advertising", () => {
    document.cookie = `${ACCOUNT_CREATED_COOKIE}=email; Path=/`;

    render(<AdsConversionTracker />);

    expect(conversions()[0][2]).toMatchObject({
      transaction_id: "user-1",
      user_data: { email: "ada@example.com" },
    });
  });

  it("drops the identifiers while the banner is unanswered", () => {
    consent.clear();
    document.cookie = `${ACCOUNT_CREATED_COOKIE}=email; Path=/`;

    render(<AdsConversionTracker />);

    expect(conversions()).toEqual([
      ["event", "conversion", { send_to: "AW-123/SU" }],
    ]);
  });

  it("sends a page_view to the Ads tag on client-side navigation only", () => {
    const { rerender } = render(<AdsConversionTracker />);
    expect(pageViews()).toEqual([]);

    pathname = "/library";
    rerender(<AdsConversionTracker />);

    expect(pageViews()).toEqual([
      ["event", "page_view", { send_to: "AW-123", page_path: "/library" }],
    ]);
  });
});
