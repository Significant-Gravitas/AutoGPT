import { describe, expect, test } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";

import type { GraphTriggerInfo } from "@/app/api/__generated__/models/graphTriggerInfo";
import type { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";
import type { Webhook } from "@/app/api/__generated__/models/webhook";
import { WebhookTriggerCard } from "./WebhookTriggerCard";

const webhook = {
  id: "webhook-1",
  user_id: "user-1",
  provider: "github",
  credentials_id: "cred-1",
  webhook_type: "repo",
  resource: "owner/repo",
  events: ["push"],
  secret: "shh",
  provider_webhook_id: "prov-1",
  url: "https://example.com/webhook",
} satisfies Webhook;

function makeTemplate(overrides: Partial<LibraryAgentPreset> = {}) {
  return {
    id: "template-1",
    user_id: "user-1",
    graph_id: "graph-1",
    graph_version: 1,
    name: "Template",
    description: "",
    inputs: {},
    credentials: {},
    is_active: true,
    webhook_id: "webhook-1",
    webhook,
    created_at: new Date("2026-01-01T00:00:00.000Z"),
    updated_at: new Date("2026-01-01T00:00:00.000Z"),
    ...overrides,
  } as LibraryAgentPreset;
}

// credentials_input_name truthy so the component falls through to the
// status-dependent description text rather than the webhook-URL block.
const triggerSetupInfo = {
  provider: "github",
  config_schema: {},
  credentials_input_name: "credentials",
} as GraphTriggerInfo;

describe("WebhookTriggerCard", () => {
  test("payment-lapsed template shows the paused badge and explanation", async () => {
    render(
      <WebhookTriggerCard
        template={makeTemplate({
          is_active: false,
          deactivation_reason: "PAYMENT_LAPSED",
        })}
        triggerSetupInfo={triggerSetupInfo}
      />,
    );

    expect(await screen.findByText("Paused — payment required")).toBeDefined();
    expect(
      screen.getByText(/paused because your payment lapsed/i),
    ).toBeDefined();
  });

  test("active template shows the active badge", async () => {
    render(
      <WebhookTriggerCard
        template={makeTemplate({ is_active: true })}
        triggerSetupInfo={triggerSetupInfo}
      />,
    );

    expect(await screen.findByText("Active")).toBeDefined();
  });

  test("user-disabled template (no reason) shows inactive, not paused", async () => {
    render(
      <WebhookTriggerCard
        template={makeTemplate({ is_active: false })}
        triggerSetupInfo={triggerSetupInfo}
      />,
    );

    expect(await screen.findByText("Inactive")).toBeDefined();
    expect(screen.queryByText("Paused — payment required")).toBeNull();
  });

  test("template without a webhook shows broken", async () => {
    render(
      <WebhookTriggerCard
        template={makeTemplate({ webhook: null, webhook_id: undefined })}
        triggerSetupInfo={triggerSetupInfo}
      />,
    );

    expect(await screen.findByText("Broken")).toBeDefined();
  });
});
