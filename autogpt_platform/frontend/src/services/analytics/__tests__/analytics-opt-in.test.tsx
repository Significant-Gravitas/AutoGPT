import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { consent } from "@/services/consent/cookies";
import { SetupAnalytics } from "../index";
import { VercelAnalyticsWrapper } from "../VercelAnalyticsWrapper";

vi.mock("@/services/consent/cookies", () => ({
  consent: { hasConsentFor: vi.fn(() => true) },
}));

vi.mock("next/navigation", () => ({
  usePathname: () => "/library",
}));

vi.mock("next/script", () => ({
  default: ({ id, src }: { id?: string; src?: string }) => (
    <script data-testid={id || "external-analytics"} data-src={src} />
  ),
}));

vi.mock("@vercel/analytics/next", () => ({
  Analytics: () => <div data-testid="vercel-analytics" />,
}));

vi.mock("@vercel/speed-insights/next", () => ({
  SpeedInsights: () => <div data-testid="speed-insights" />,
}));

describe("self-hosted analytics opt-in", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(consent.hasConsentFor).mockReturnValue(true);
  });

  it("does not render analytics integrations when the operator disabled them", async () => {
    render(
      <>
        <SetupAnalytics
          dataFastEnabled
          enabled={false}
          ga={{ gaId: "G-AUTOGPT" }}
        />
        <VercelAnalyticsWrapper enabled={false} />
      </>,
    );

    await waitFor(() => {
      expect(screen.queryByTestId("_custom-ga-init")).toBeNull();
      expect(screen.queryByTestId("_custom-ga")).toBeNull();
      expect(screen.queryByTestId("external-analytics")).toBeNull();
      expect(screen.queryByTestId("vercel-analytics")).toBeNull();
      expect(screen.queryByTestId("speed-insights")).toBeNull();
    });
  });

  it("renders consented analytics on the hosted platform", async () => {
    render(
      <>
        <SetupAnalytics dataFastEnabled enabled ga={{ gaId: "G-AUTOGPT" }} />
        <VercelAnalyticsWrapper enabled />
      </>,
    );

    await waitFor(() => {
      expect(screen.getByTestId("_custom-ga-init")).toBeDefined();
      expect(screen.getByTestId("_custom-ga")).toBeDefined();
      expect(
        screen.getByTestId("external-analytics").getAttribute("data-src"),
      ).toBe("https://datafa.st/js/script.js");
      expect(screen.getByTestId("vercel-analytics")).toBeDefined();
      expect(screen.getByTestId("speed-insights")).toBeDefined();
    });
  });

  it("renders an operator-configured GA property on a self-hosted domain", async () => {
    render(
      <>
        <SetupAnalytics
          dataFastEnabled={false}
          enabled
          ga={{ gaId: "G-OPERATOR" }}
        />
        <VercelAnalyticsWrapper enabled={false} />
      </>,
    );

    await waitFor(() => {
      expect(screen.getByTestId("_custom-ga-init")).toBeDefined();
      expect(screen.getByTestId("_custom-ga")).toBeDefined();
      expect(screen.queryByTestId("external-analytics")).toBeNull();
      expect(screen.queryByTestId("vercel-analytics")).toBeNull();
      expect(screen.queryByTestId("speed-insights")).toBeNull();
    });
  });

  it("does not render analytics when browser consent is denied", async () => {
    vi.mocked(consent.hasConsentFor).mockReturnValue(false);

    render(
      <SetupAnalytics dataFastEnabled enabled ga={{ gaId: "G-AUTOGPT" }} />,
    );

    await waitFor(() => {
      expect(screen.queryByTestId("_custom-ga-init")).toBeNull();
      expect(screen.queryByTestId("_custom-ga")).toBeNull();
      expect(screen.queryByTestId("external-analytics")).toBeNull();
    });
  });
});
