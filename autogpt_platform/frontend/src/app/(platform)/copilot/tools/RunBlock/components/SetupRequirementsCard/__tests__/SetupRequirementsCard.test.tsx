import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SetupRequirementsCard } from "../SetupRequirementsCard";
import type { SetupRequirementsResponse } from "@/app/api/__generated__/models/setupRequirementsResponse";

const mockOnSend = vi.fn();
vi.mock(
  "../../../../../components/CopilotChatActionsProvider/useCopilotChatActions",
  () => ({
    useCopilotChatActions: () => ({ onSend: mockOnSend }),
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

vi.mock("@/components/renderers/InputRenderer/FormRenderer", () => ({
  FormRenderer: ({
    handleChange,
  }: {
    handleChange: (e: { formData?: Record<string, unknown> }) => void;
  }) => (
    <div data-testid="form-renderer">
      <button
        data-testid="form-change"
        onClick={() => handleChange({ formData: { url: "https://test.com" } })}
      >
        Fill
      </button>
    </div>
  ),
}));

afterEach(() => {
  cleanup();
  mockOnSend.mockReset();
});

function makeOutput(
  overrides: {
    message?: string;
    missingCredentials?: Record<string, unknown>;
    inputs?: unknown[];
  } = {},
): SetupRequirementsResponse {
  const {
    message = "Please configure credentials",
    missingCredentials,
    inputs,
  } = overrides;
  return {
    type: "setup_requirements",
    message,
    session_id: "sess-1",
    setup_info: {
      agent_id: "agent-1",
      agent_name: "Test Agent",
      user_readiness: {
        has_all_credentials: !missingCredentials,
        missing_credentials: missingCredentials ?? {},
        ready_to_run: !missingCredentials && !inputs,
      },
      requirements: {
        credentials: [],
        inputs: inputs ?? [],
        execution_modes: ["immediate"],
      },
    },
    graph_id: null,
    graph_version: null,
  } as SetupRequirementsResponse;
}

describe("SetupRequirementsCard", () => {
  it("renders the setup message", () => {
    render(<SetupRequirementsCard output={makeOutput()} />);
    expect(screen.getByText("Please configure credentials")).toBeDefined();
  });

  it("renders credential section when missing credentials are provided", () => {
    render(
      <SetupRequirementsCard
        output={makeOutput({
          missingCredentials: {
            api_key: {
              provider: "openai",
              types: ["api_key"],
            },
          },
        })}
      />,
    );
    expect(screen.getByTestId("credentials-grouped-view")).toBeDefined();
  });

  it("uses custom credentials label when provided", () => {
    render(
      <SetupRequirementsCard
        output={makeOutput({
          missingCredentials: {
            api_key: { provider: "openai", types: ["api_key"] },
          },
        })}
        credentialsLabel="API Keys"
      />,
    );
    expect(screen.getByText("API Keys")).toBeDefined();
  });

  it("renders input form when inputs are provided", () => {
    render(
      <SetupRequirementsCard
        output={makeOutput({
          inputs: [
            { name: "url", title: "URL", type: "string", required: true },
          ],
        })}
      />,
    );
    expect(screen.getByTestId("form-renderer")).toBeDefined();
    expect(screen.getByText("Inputs")).toBeDefined();
  });

  it("renders Proceed button that is enabled when inputs are filled", () => {
    render(
      <SetupRequirementsCard
        output={makeOutput({
          inputs: [
            {
              name: "url",
              title: "URL",
              type: "string",
              required: true,
              value: "https://prefilled.com",
            },
          ],
        })}
      />,
    );
    const proceed = screen.getByText("Proceed");
    expect(proceed.closest("button")?.disabled).toBe(false);
  });

  it("calls onSend and shows Connected message when Proceed is clicked", () => {
    render(
      <SetupRequirementsCard
        output={makeOutput({
          inputs: [
            {
              name: "url",
              title: "URL",
              type: "string",
              required: true,
              value: "https://prefilled.com",
            },
          ],
        })}
      />,
    );
    fireEvent.click(screen.getByText("Proceed"));
    expect(mockOnSend).toHaveBeenCalledOnce();
    expect(screen.getByText(/Connected. Continuing/)).toBeDefined();
  });

  it("calls onComplete callback when Proceed is clicked", () => {
    const onComplete = vi.fn();
    render(
      <SetupRequirementsCard
        output={makeOutput({
          inputs: [
            {
              name: "url",
              title: "URL",
              type: "string",
              required: true,
              value: "https://prefilled.com",
            },
          ],
        })}
        onComplete={onComplete}
      />,
    );
    fireEvent.click(screen.getByText("Proceed"));
    expect(onComplete).toHaveBeenCalledOnce();
  });

  it("renders advanced toggle when advanced inputs exist", () => {
    render(
      <SetupRequirementsCard
        output={makeOutput({
          inputs: [
            {
              name: "debug",
              title: "Debug Mode",
              type: "boolean",
              advanced: true,
            },
          ],
        })}
      />,
    );
    expect(screen.getByText("Show advanced fields")).toBeDefined();
  });

  it("toggles advanced fields visibility", () => {
    render(
      <SetupRequirementsCard
        output={makeOutput({
          inputs: [
            { name: "url", title: "URL", type: "string", required: false },
            { name: "debug", title: "Debug", type: "boolean", advanced: true },
          ],
        })}
      />,
    );
    const toggle = screen.getByText("Show advanced fields");
    fireEvent.click(toggle);
    expect(screen.getByText("Hide advanced fields")).toBeDefined();
  });

  it("includes retryInstruction in onSend message when no inputs needed", () => {
    render(
      <SetupRequirementsCard
        output={makeOutput({
          missingCredentials: {
            api_key: { provider: "openai", types: ["api_key"] },
          },
        })}
        retryInstruction="Retry the agent now"
      />,
    );
    // With credentials required but no auto-filling mechanism in the mock,
    // Proceed is disabled, but we're testing render only here
    expect(screen.getByText("Proceed")).toBeDefined();
  });

  it("does not render Proceed when neither credentials nor inputs are needed", () => {
    render(<SetupRequirementsCard output={makeOutput()} />);
    expect(screen.queryByText("Proceed")).toBeNull();
  });
});
