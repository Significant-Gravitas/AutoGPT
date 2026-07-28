import type { GraphTriggerInfo } from "@/app/api/__generated__/models/graphTriggerInfo";
import type { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";
import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, test } from "vitest";
import { WebhookTriggerSection } from "../WebhookTriggerSection";

function makePreset(
  overrides: Partial<LibraryAgentPreset>,
): LibraryAgentPreset {
  return {
    id: "preset-1",
    user_id: "user-1",
    graph_id: "graph-1",
    graph_version: 1,
    name: "My trigger",
    description: "",
    inputs: {},
    credentials: {},
    is_active: true,
    created_at: new Date(),
    updated_at: new Date(),
    webhook_id: "wh-1",
    webhook: {
      id: "wh-1",
      user_id: "user-1",
      url: "https://example.com/hooks/wh-1",
      provider: "generic_webhook",
      credentials_id: "",
      webhook_type: "manual",
      resource: "",
      events: [],
      config: {},
      secret: "",
      provider_webhook_id: "",
    } as LibraryAgentPreset["webhook"],
    ...overrides,
  } as LibraryAgentPreset;
}

const triggerSetupInfo = {
  provider: "generic_webhook",
  config_schema: {},
  credentials_input_name: null,
} as unknown as GraphTriggerInfo;

describe("WebhookTriggerSection", () => {
  test("active preset shows Active status and webhook URL", () => {
    render(
      <WebhookTriggerSection
        preset={makePreset({})}
        triggerSetupInfo={triggerSetupInfo}
      />,
    );
    expect(screen.getByText("Active")).toBeDefined();
    expect(screen.getByText("https://example.com/hooks/wh-1")).toBeDefined();
  });

  test("payment-lapse deactivated preset shows paused status, not the ready copy", () => {
    render(
      <WebhookTriggerSection
        preset={makePreset({
          is_active: false,
          deactivation_reason: "PAYMENT_LAPSED",
        })}
        triggerSetupInfo={triggerSetupInfo}
      />,
    );
    expect(screen.getByText("Paused — payment required")).toBeDefined();
    expect(screen.queryByText("https://example.com/hooks/wh-1")).toBeNull();
    expect(
      screen.getByText(/paused because your payment lapsed/i),
    ).toBeDefined();
  });

  test("user-deactivated preset shows Inactive, not payment copy", () => {
    render(
      <WebhookTriggerSection
        preset={makePreset({ is_active: false })}
        triggerSetupInfo={triggerSetupInfo}
      />,
    );
    expect(screen.getByText("Inactive")).toBeDefined();
    expect(screen.queryByText(/payment/i)).toBeNull();
  });
});
