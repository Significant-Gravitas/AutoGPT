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
import { VercelAnalyticsWrapper } from "../VercelAnalyticsWrapper";

vi.mock("next/script", () => ({
  default: ({
    id,
    src,
    strategy,
    dangerouslySetInnerHTML: _inline,
    onLoad: _onLoad,
    ...attributes
  }: {
    id?: string;
    src?: string;
    strategy?: string;
    dangerouslySetInnerHTML?: unknown;
    onLoad?: unknown;
  }) => (
    <div
      data-testid={`script:${id ?? src}`}
      data-src={src}
      data-strategy={strategy}
      {...attributes}
    />
  ),
}));

vi.mock("@vercel/analytics/next", () => ({
  Analytics: () => <div data-testid="vercel-analytics" />,
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
        <VercelAnalyticsWrapper />
      </>,
    );

    expect(queryScript("_custom-ga")).toBeNull();
    expect(queryScript(DATAFAST_SRC)).toBeNull();
    expect(screen.queryByTestId("vercel-analytics")).toBeNull();
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
      },
    ]);
    delete (window as unknown as { dataLayer?: unknown }).dataLayer;
  });

  it("starts DataFast and Vercel Analytics when the visitor accepts, without a reload", () => {
    installCookiebot();
    render(
      <>
        <SetupAnalytics host="platform.agpt.co" ga={GA} />
        <VercelAnalyticsWrapper />
      </>,
    );
    expect(queryScript(DATAFAST_SRC)).toBeNull();
    expect(screen.queryByTestId("vercel-analytics")).toBeNull();

    act(() => answerCookiebot({ statistics: true }));

    expect(queryScript(DATAFAST_SRC)).not.toBeNull();
    expect(screen.getByTestId("vercel-analytics")).toBeDefined();
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

  it("stops rendering Vercel Analytics when the visitor declines", () => {
    installCookiebot({ statistics: true });
    render(<VercelAnalyticsWrapper />);
    expect(screen.getByTestId("vercel-analytics")).toBeDefined();

    act(() => answerCookiebot({}));

    expect(screen.queryByTestId("vercel-analytics")).toBeNull();
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

  it("does not reload when the visitor grants more, or repeats the same answer", () => {
    installCookiebot();
    render(<ConsentWithdrawalReload />);

    act(() => answerCookiebot({ statistics: true }));
    act(() => answerCookiebot({ statistics: true }));
    act(() => answerCookiebot({ statistics: true, marketing: true }));

    expect(reload).not.toHaveBeenCalled();
  });
});
