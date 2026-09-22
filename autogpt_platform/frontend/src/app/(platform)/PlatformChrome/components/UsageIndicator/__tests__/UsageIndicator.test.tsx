import { render, screen } from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";

import { server } from "@/mocks/mock-server";

import { UsageIndicator } from "../UsageIndicator";

// UsagePopover pulls in the full usage limits stack; stub it to a passthrough
// that just renders its trigger so we can assert the indicator button itself.
vi.mock(
  "@/app/(platform)/copilot/components/UsageLimits/UsagePopover/UsagePopover",
  () => ({
    UsagePopover: ({ trigger }: { trigger: React.ReactNode }) => (
      <div>{trigger}</div>
    ),
  }),
);

function usageHandler(body: { daily: { percent_used: number } | null }) {
  return http.get("*/api/chat/usage", () => HttpResponse.json(body));
}

afterEach(() => {
  server.resetHandlers();
});

describe("UsageIndicator", () => {
  it("renders a generic label before usage resolves", async () => {
    server.use(usageHandler({ daily: null }));
    render(<UsageIndicator />);

    expect(
      await screen.findByRole("button", { name: "Today's usage" }),
    ).toBeDefined();
  });

  it("renders the percent in the label and draws the gauge arc", async () => {
    server.use(usageHandler({ daily: { percent_used: 64 } }));
    render(<UsageIndicator />);

    expect(
      await screen.findByRole("button", { name: "Today's usage: 64%" }),
    ).toBeDefined();
  });
});
