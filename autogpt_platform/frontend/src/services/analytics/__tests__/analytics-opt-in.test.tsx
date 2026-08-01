import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { SetupAnalytics } from "../index";
import { VercelAnalyticsWrapper } from "../VercelAnalyticsWrapper";

vi.mock("@/services/consent/cookies", () => ({
  consent: { hasConsentFor: vi.fn(() => true) },
}));

vi.mock("@/services/environment", () => ({
  environment: { isLocal: vi.fn(() => true) },
}));

vi.mock("next/navigation", () => ({
  usePathname: () => "/library",
}));

vi.mock("next/script", () => ({
  default: ({ id }: { id?: string }) => <script data-testid={id} />,
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
  });

  it("does not render analytics integrations when the operator disabled them", async () => {
    render(
      <>
        <SetupAnalytics
          enabled={false}
          host="localhost:3000"
          ga={{ gaId: "G-AUTOGPT" }}
        />
        <VercelAnalyticsWrapper enabled={false} />
      </>,
    );

    await waitFor(() => {
      expect(screen.queryByTestId("_custom-ga")).toBeNull();
      expect(screen.queryByTestId("vercel-analytics")).toBeNull();
      expect(screen.queryByTestId("speed-insights")).toBeNull();
    });
  });
});
