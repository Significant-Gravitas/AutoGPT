import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  answerCookiebot,
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";
import { ConsentWithdrawalReload } from "@/services/consent/ConsentWithdrawalReload";
import { SetupAnalytics } from "../index";
import { VercelSpeedInsights } from "../VercelSpeedInsights";

vi.mock("next/script", () => ({
  default: ({
    id,
    src,
    strategy,
    dangerouslySetInnerHTML: inline,
    onLoad: _onLoad,
    ...attributes
  }: {
    id?: string;
    src?: string;
    strategy?: string;
    dangerouslySetInnerHTML?: { __html?: string };
    onLoad?: unknown;
  }) => (
    <div
      data-testid={`script:${id ?? src}`}
      data-src={src}
      data-strategy={strategy}
      data-inline={inline?.__html}
      {...attributes}
    />
  ),
}));

vi.mock("@vercel/speed-insights/next", () => ({
  SpeedInsights: () => <div data-testid="vercel-speed-insights" />,
}));

const GA = { gaId: "G-TEST" };
const DATAFAST_SRC = "https://datafa.st/js/script.js";

function queryScript(name: string) {
  return screen.queryByTestId(`script:${name}`);
}

beforeEach(() => {
  vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "CLOUD");
  vi.stubEnv("NEXT_PUBLIC_COOKIEBOT_CBID", "");
});

afterEach(() => {
  removeCookiebot();
  vi.unstubAllEnvs();
});

describe("without NEXT_PUBLIC_COOKIEBOT_CBID", () => {
  it("loads no analytics, even for a browser holding an old answer", () => {
    installCookiebot({ statistics: true, marketing: true });

    render(
      <>
        <SetupAnalytics host="platform.agpt.co" ga={GA} />
        <VercelSpeedInsights />
      </>,
    );

    expect(queryScript("_custom-ga")).toBeNull();
    expect(queryScript(DATAFAST_SRC)).toBeNull();
    expect(screen.queryByTestId("vercel-speed-insights")).toBeNull();
  });

  it("keeps the Google tag off locally even for an opted-in browser", () => {
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "LOCAL");
    installCookiebot({ statistics: true });

    render(<SetupAnalytics host="localhost:3000" ga={GA} />);

    expect(queryScript("_custom-ga")).toBeNull();
  });
});

describe("with NEXT_PUBLIC_COOKIEBOT_CBID", () => {
  beforeEach(() => {
    configureCookiebot();
  });

  it("loads the Google tag on production before the visitor answers", () => {
    installCookiebot();

    render(<SetupAnalytics host="platform.agpt.co" ga={GA} />);

    expect(queryScript("_custom-ga")).not.toBeNull();
    expect(queryScript(DATAFAST_SRC)).toBeNull();
  });

  it("sends the answer to the Google tag as a Consent Mode update", () => {
    installCookiebot();
    render(<SetupAnalytics host="platform.agpt.co" ga={GA} />);

    act(() => answerCookiebot({ statistics: true }));

    const dataLayer = (window as unknown as { dataLayer?: IArguments[] })
      .dataLayer;
    expect(dataLayer?.map((entry) => Array.from(entry))).toContainEqual([
      "consent",
      "update",
      {
        analytics_storage: "granted",
        ad_storage: "denied",
        ad_user_data: "denied",
        ad_personalization: "denied",
        functionality_storage: "denied",
        personalization_storage: "denied",
      },
    ]);
    delete (window as unknown as { dataLayer?: unknown }).dataLayer;
  });

  it("starts DataFast and Vercel Speed Insights when the visitor accepts, without a reload", () => {
    installCookiebot();
    render(
      <>
        <SetupAnalytics host="platform.agpt.co" ga={GA} />
        <VercelSpeedInsights />
      </>,
    );
    expect(queryScript(DATAFAST_SRC)).toBeNull();
    expect(screen.queryByTestId("vercel-speed-insights")).toBeNull();

    act(() => answerCookiebot({ statistics: true }));

    expect(queryScript(DATAFAST_SRC)).not.toBeNull();
    expect(screen.getByTestId("vercel-speed-insights")).toBeDefined();
  });

  it("loads the Google tag locally once the visitor accepts statistics", () => {
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "LOCAL");
    installCookiebot();
    render(<SetupAnalytics host="localhost:3000" ga={GA} />);
    expect(queryScript("_custom-ga")).toBeNull();

    act(() => answerCookiebot({ marketing: true }));
    expect(queryScript("_custom-ga")).toBeNull();

    act(() => answerCookiebot({ statistics: true }));
    expect(queryScript("_custom-ga")).not.toBeNull();
  });

  it("stops rendering Vercel Speed Insights when the visitor declines", () => {
    installCookiebot({ statistics: true });
    render(<VercelSpeedInsights />);
    expect(screen.getByTestId("vercel-speed-insights")).toBeDefined();

    act(() => answerCookiebot({}));

    expect(screen.queryByTestId("vercel-speed-insights")).toBeNull();
  });
});

describe("Google tag IDs", () => {
  beforeEach(() => {
    configureCookiebot();
    installCookiebot({ statistics: true, marketing: true });
  });

  it("loads no Google tag when neither a GA nor a Google Ads ID is set", () => {
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_ADS_ID", "");

    render(<SetupAnalytics host="platform.agpt.co" ga={{ gaId: "" }} />);

    expect(queryScript("_custom-ga")).toBeNull();
    expect(queryScript("_custom-ga-init")).toBeNull();
  });

  it("configures only Google Ads when the GA ID is not set", () => {
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_ADS_ID", "AW-TEST");

    render(<SetupAnalytics host="platform.agpt.co" ga={{ gaId: "" }} />);

    const init = queryScript("_custom-ga-init")?.getAttribute("data-inline");
    expect(queryScript("_custom-ga")).not.toBeNull();
    expect(init).toContain(`gtag('config', "AW-TEST"`);
    expect(init).not.toContain(`gtag('config', ""`);
  });

  it("configures the GA property it is given", () => {
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_ADS_ID", "");

    render(<SetupAnalytics host="platform.agpt.co" ga={GA} />);

    const init = queryScript("_custom-ga-init")?.getAttribute("data-inline");
    expect(init).toContain(`gtag('config', "G-TEST"`);
  });
});

describe("ConsentWithdrawalReload", () => {
  const reload = vi.fn<() => void>();

  beforeEach(() => {
    configureCookiebot();
    reload.mockReset();
    vi.spyOn(window.location, "reload").mockImplementation(() => reload());
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("reloads when a granted category is taken back", () => {
    installCookiebot({ statistics: true, marketing: true });
    render(<ConsentWithdrawalReload />);

    act(() => answerCookiebot({ statistics: true }));

    expect(reload).toHaveBeenCalledOnce();
  });

  it("does not reload when Cookiebot invalidates a stale granting cookie, only on the visitor's reply", () => {
    const stale =
      "{stamp:'abc',necessary:true,preferences:false,statistics:true,marketing:false,method:'explicit',ver:1,utc:1,region:'de'}";
    document.cookie = `CookieConsent=${encodeURIComponent(stale)}; Path=/`;
    render(<ConsentWithdrawalReload />);

    act(() => {
      installCookiebot();
      window.dispatchEvent(new Event("CookiebotOnLoad"));
    });

    expect(reload).not.toHaveBeenCalled();

    act(() => answerCookiebot({}));

    expect(reload).toHaveBeenCalledOnce();
  });

  it("does not reload when only preferences are taken back", () => {
    installCookiebot({ statistics: true, preferences: true });
    render(<ConsentWithdrawalReload />);

    act(() => answerCookiebot({ statistics: true }));

    expect(reload).not.toHaveBeenCalled();
  });

  it("does not reload when the visitor grants more, or repeats the same answer", () => {
    installCookiebot();
    render(<ConsentWithdrawalReload />);

    act(() => answerCookiebot({ statistics: true }));
    act(() => answerCookiebot({ statistics: true }));
    act(() => answerCookiebot({ statistics: true, marketing: true }));

    expect(reload).not.toHaveBeenCalled();
  });
});
