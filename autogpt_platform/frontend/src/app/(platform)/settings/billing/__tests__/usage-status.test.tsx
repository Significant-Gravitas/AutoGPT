import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, expect, it } from "vitest";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import {
  setTrialUser,
  trialResponse,
} from "@/tests/integrations/trial-fixtures";
import { AutopilotUsageCard } from "../components/SubscriptionTab/AutopilotUsageCard/AutopilotUsageCard";

beforeEach(() => setTrialUser());
afterEach(() => setTrialUser(null));

it.each([
  [0, "Plenty left"],
  [79, "Plenty left"],
  [80, "Running low"],
  [99, "Running low"],
  [100, "Allowance reached"],
])(
  "reports the featured trial allowance at %s%% as %s",
  async (percent, status) => {
    server.use(
      http.get("*/api/credits/subscription", () =>
        HttpResponse.json({ tier: "TRIAL", monthly_cost: 0 }),
      ),
      http.get("*/api/credits/trial", () =>
        HttpResponse.json(
          trialResponse({
            usage_policy: "lifetime",
            allowance_used_percent: percent,
          }),
        ),
      ),
      http.get("*/api/chat/usage", () =>
        HttpResponse.json({
          tier: "TRIAL",
          daily: { percent_used: 5, resets_at: null },
          weekly: { percent_used: 10, resets_at: null },
        }),
      ),
    );
    render(<AutopilotUsageCard />);
    expect(await screen.findByText(status)).toBeDefined();
    expect(screen.queryByText("On track")).toBeNull();
  },
);
