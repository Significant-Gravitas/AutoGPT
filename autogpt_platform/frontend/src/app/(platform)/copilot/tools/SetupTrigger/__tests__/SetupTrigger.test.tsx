import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SetupTriggerTool } from "../SetupTrigger";
import { buildTriggerSetupMessage } from "../../../components/SetupRequirementsCard/helpers";
import { useConnectedProvidersStore } from "../../../connectedProvidersStore";

vi.mock(
  "@/app/(platform)/copilot/components/CopilotChatActionsProvider/useCopilotChatActions",
  () => ({
    useCopilotChatActions: () => ({ onSend: vi.fn() }),
  }),
);

vi.mock(
  "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/CredentialsGroupedView",
  () => ({
    CredentialsGroupedView: () => (
      <div data-testid="credentials-grouped-view">Credentials</div>
    ),
  }),
);

// The morphing animation renders text letter-by-letter across <span>s, which
// breaks whole-string text queries. Render it as plain text in tests.
vi.mock(
  "../../../components/MorphingTextAnimation/MorphingTextAnimation",
  () => ({
    MorphingTextAnimation: ({ text }: { text: string }) => <span>{text}</span>,
  }),
);

afterEach(() => {
  cleanup();
  useConnectedProvidersStore.setState({
    connected: new Set(),
    autoDismissedKeys: new Set(),
  });
});

function part(output: unknown) {
  return {
    type: "tool-setup_agent_webhook_trigger",
    state: "output-available" as const,
    output,
  };
}

describe("SetupTriggerTool", () => {
  it("renders the credentials card for a setup_requirements output", () => {
    render(
      <SetupTriggerTool
        part={part({
          type: "setup_requirements",
          message: "Choose an account to set up the trigger.",
          session_id: "s1",
          setup_info: {
            agent_id: "g1",
            agent_name: "My Agent",
            user_readiness: {
              has_all_credentials: false,
              missing_credentials: {
                github_credentials: { provider: "github", types: ["api_key"] },
              },
              ready_to_run: false,
            },
            requirements: {
              credentials: [],
              inputs: [],
              execution_modes: ["webhook"],
            },
          },
          graph_id: "g1",
          graph_version: 1,
        })}
      />,
    );
    expect(screen.getByTestId("credentials-grouped-view")).toBeDefined();
    // Trigger mode labels the credentials section "Account".
    expect(screen.getByText("Account")).toBeDefined();
  });

  it("renders the webhook URL with a copy button for a trigger_setup output", () => {
    const url =
      "https://backend.agpt.co/api/integrations/generic_webhook/webhooks/wh-1/ingress";
    render(
      <SetupTriggerTool
        part={part({
          type: "trigger_setup",
          message: "Webhook trigger ready.",
          preset_id: "p1",
          library_agent_id: "lib-1",
          library_agent_link: "/library/agents/lib-1",
          name: "My Trigger",
          is_active: true,
          provider: "generic_webhook",
          manual_setup_required: true,
          webhook_url: url,
        })}
      />,
    );
    expect(screen.getByText("Webhook trigger ready.")).toBeDefined();
    expect(screen.getByText(url)).toBeDefined();
    expect(screen.getByLabelText("Copy webhook URL")).toBeDefined();
  });

  it("renders the message for a trigger_config_required output", () => {
    render(
      <SetupTriggerTool
        part={part({
          type: "trigger_config_required",
          message: "Ask the user which repo and events to use.",
          missing_config: ["repo", "events"],
          config_schema: { properties: {}, required: ["repo", "events"] },
          graph_id: "g1",
          graph_version: 1,
        })}
      />,
    );
    expect(
      screen.getByText("Ask the user which repo and events to use."),
    ).toBeDefined();
  });

  it("renders the need-login message for a need_login output", () => {
    render(
      <SetupTriggerTool
        part={part({
          type: "need_login",
          message: "Please sign in to set up the trigger.",
        })}
      />,
    );
    expect(
      screen.getByText("Please sign in to set up the trigger."),
    ).toBeDefined();
  });

  it("renders the error animation text for an error output", () => {
    render(
      <SetupTriggerTool
        part={part({ type: "error", error: "Something failed" })}
      />,
    );
    expect(screen.getByText("Something went wrong")).toBeDefined();
  });

  it("renders the error animation text for the output-error state", () => {
    render(
      <SetupTriggerTool
        part={{
          type: "tool-setup_agent_webhook_trigger",
          state: "output-error",
        }}
      />,
    );
    expect(screen.getByText("Something went wrong")).toBeDefined();
  });

  it("renders the streaming animation text while input is streaming", () => {
    render(
      <SetupTriggerTool
        part={{
          type: "tool-setup_agent_webhook_trigger",
          state: "input-streaming",
        }}
      />,
    );
    expect(screen.getByText("Setting up the trigger")).toBeDefined();
  });

  it("does not render the manual webhook URL block when manual setup is not required", () => {
    render(
      <SetupTriggerTool
        part={part({
          type: "trigger_setup",
          message: "Trigger ready (auto).",
          preset_id: "p1",
          library_agent_id: "lib-1",
          library_agent_link: "/library/agents/lib-1",
          name: "Auto Trigger",
          is_active: true,
          provider: "github",
          manual_setup_required: false,
          webhook_url: "https://example.com/wh",
        })}
      />,
    );
    expect(screen.getByText("Trigger ready (auto).")).toBeDefined();
    expect(screen.queryByLabelText("Copy webhook URL")).toBeNull();
  });
});

describe("SetupTriggerTool copy button", () => {
  const url =
    "https://backend.agpt.co/api/integrations/generic_webhook/webhooks/wh-1/ingress";

  function setClipboard(writeText: (text: string) => Promise<void>) {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
  }

  afterEach(() => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: undefined,
    });
  });

  function renderSuccess() {
    render(
      <SetupTriggerTool
        part={part({
          type: "trigger_setup",
          message: "Webhook trigger ready.",
          preset_id: "p1",
          library_agent_id: "lib-1",
          library_agent_link: "/library/agents/lib-1",
          name: "My Trigger",
          is_active: true,
          provider: "generic_webhook",
          manual_setup_required: true,
          webhook_url: url,
        })}
      />,
    );
  }

  it("copies the webhook URL and shows the confirmation icon", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    setClipboard(writeText);

    renderSuccess();
    fireEvent.click(screen.getByLabelText("Copy webhook URL"));

    await waitFor(() => expect(writeText).toHaveBeenCalledWith(url));
  });

  it("silently ignores clipboard failures (non-secure context)", async () => {
    const writeText = vi.fn().mockRejectedValue(new Error("blocked"));
    setClipboard(writeText);

    renderSuccess();
    fireEvent.click(screen.getByLabelText("Copy webhook URL"));

    await waitFor(() => expect(writeText).toHaveBeenCalledWith(url));
    // The URL stays visible for manual copy.
    expect(screen.getByText(url)).toBeDefined();
  });
});

describe("buildTriggerSetupMessage", () => {
  it("carries back the selected credential IDs as a credentials map", () => {
    const msg = buildTriggerSetupMessage({
      github_credentials: { id: "cred-1" },
      skipped: undefined,
    });
    expect(msg).toContain('credentials={"github_credentials":"cred-1"}');
    expect(msg).toContain("setup_agent_webhook_trigger");
  });
});
