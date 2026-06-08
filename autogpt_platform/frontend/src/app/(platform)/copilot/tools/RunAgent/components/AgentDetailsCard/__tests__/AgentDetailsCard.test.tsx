import type { AgentDetailsResponse } from "@/app/api/__generated__/models/agentDetailsResponse";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AgentDetailsCard } from "../AgentDetailsCard";

const onSend = vi.fn();

vi.mock(
  "@/app/(platform)/copilot/components/CopilotChatActionsProvider/useCopilotChatActions",
  () => ({
    useCopilotChatActions: () => ({ onSend }),
  }),
);

afterEach(() => {
  cleanup();
  onSend.mockReset();
});

function webhookOutput(): AgentDetailsResponse {
  return {
    type: "agent_details",
    message: "This agent runs on a webhook trigger.",
    user_authenticated: true,
    agent: {
      id: "g-wh",
      name: "PR Notifier",
      description: "Notifies on PRs",
      inputs: {},
      execution_options: { manual: false, scheduled: false, webhook: true },
      trigger_info: {
        provider: "github",
        config_schema: {
          type: "object",
          properties: { repo: { type: "string" } },
          required: ["repo"],
        },
        credentials_input_name: null,
      },
    },
  } as AgentDetailsResponse;
}

function nonWebhookOutput(): AgentDetailsResponse {
  return {
    type: "agent_details",
    message: "Ready to run.",
    user_authenticated: true,
    agent: {
      id: "g-run",
      name: "Summariser",
      description: "Summarises text",
      // No configurable inputs -> the run path renders a Proceed button
      // (avoids FormRenderer, which needs app-level providers).
      inputs: {},
      execution_options: { manual: true, scheduled: true, webhook: false },
    },
  } as AgentDetailsResponse;
}

describe("AgentDetailsCard", () => {
  it("shows an informational message and no action button for a webhook-trigger agent", () => {
    render(<AgentDetailsCard output={webhookOutput()} />);

    // Informational message about the webhook trigger is shown...
    expect(screen.getByText(/webhook trigger/i)).toBeDefined();
    // ...and there is NO actionable button (no Run/Proceed/Set up trigger) that
    // could go stale after setup_agent_webhook_trigger already succeeded.
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("shows the run path (Proceed), not the webhook card, for a non-webhook agent", () => {
    render(<AgentDetailsCard output={nonWebhookOutput()} />);

    // Regular run path: the Proceed button is present and there's no
    // webhook-trigger messaging (the webhook branch must not shadow it).
    expect(screen.getByRole("button", { name: /proceed/i })).toBeDefined();
    expect(screen.queryByText(/webhook trigger/i)).toBeNull();
  });
});
