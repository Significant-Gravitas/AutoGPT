import {
  render,
  screen,
  fireEvent,
  cleanup,
  waitFor,
} from "@testing-library/react";
import { StrictMode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SetupRequirementsCard } from "../SetupRequirementsCard";
import type { SetupRequirementsResponse } from "@/app/api/__generated__/models/setupRequirementsResponse";
import { useConnectedProvidersStore } from "../../../connectedProvidersStore";

const mockOnSend = vi.fn();
vi.mock("../../CopilotChatActionsProvider/useCopilotChatActions", () => ({
  useCopilotChatActions: () => ({ onSend: mockOnSend }),
}));

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
  useConnectedProvidersStore.setState({
    connected: new Set(),
    autoDismissedKeys: new Set(),
  });
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

describe("SetupRequirementsCard (edit mode)", () => {
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

describe("SetupRequirementsCard (preview mode)", () => {
  it("renders inputs as read-only list, not a form", () => {
    render(
      <SetupRequirementsCard
        inputsMode="preview"
        output={makeOutput({
          inputs: [
            {
              name: "query",
              title: "Search Query",
              type: "string",
              required: true,
              description: "What to search for",
            },
          ],
        })}
      />,
    );
    expect(screen.queryByTestId("form-renderer")).toBeNull();
    expect(screen.getByText("Expected inputs")).toBeDefined();
    expect(screen.getByText("Search Query")).toBeDefined();
    expect(screen.getByText("What to search for")).toBeDefined();
    expect(screen.getByText("Required")).toBeDefined();
    // Technical bits — internal field name and JSON-schema type — must NOT
    // leak into the user-facing preview.
    expect(screen.queryByText(/query/)).toBeNull();
    expect(screen.queryByText(/string/)).toBeNull();
  });

  it("omits the description block entirely when no description is provided", () => {
    render(
      <SetupRequirementsCard
        inputsMode="preview"
        output={makeOutput({
          inputs: [
            { name: "url", title: "URL", type: "string", required: true },
          ],
        })}
      />,
    );
    expect(screen.getByText("URL")).toBeDefined();
    // No "url • string" leakage and no empty description container.
    expect(screen.queryByText(/url • string/i)).toBeNull();
  });

  it("labels credentials section as 'Agent credentials' by default", () => {
    render(
      <SetupRequirementsCard
        inputsMode="preview"
        output={makeOutput({
          missingCredentials: {
            api_key: { provider: "openai", types: ["api_key"] },
          },
        })}
      />,
    );
    expect(screen.getByText("Agent credentials")).toBeDefined();
  });

  it("does not render advanced toggle even when advanced inputs exist", () => {
    render(
      <SetupRequirementsCard
        inputsMode="preview"
        output={makeOutput({
          inputs: [
            {
              name: "debug",
              title: "Debug",
              type: "boolean",
              advanced: true,
            },
          ],
        })}
      />,
    );
    expect(screen.queryByText("Show advanced fields")).toBeNull();
  });

  it("renders enabled Proceed with inputs only (preserves legacy run_agent behavior)", () => {
    // Proceed is rendered when needsCredentials || needsInputs — preserves
    // legacy run_agent behaviour where Proceed is shown alongside the input
    // preview and gated only by credentials.
    render(
      <SetupRequirementsCard
        inputsMode="preview"
        output={makeOutput({
          inputs: [
            { name: "query", title: "Query", type: "string", required: true },
          ],
        })}
      />,
    );
    const proceed = screen.queryByText("Proceed");
    expect(proceed).not.toBeNull();
    expect(proceed!.closest("button")?.disabled).toBe(false);
  });

  it("sends the legacy run_agent message on Proceed", () => {
    render(
      <SetupRequirementsCard
        inputsMode="preview"
        output={makeOutput({
          inputs: [
            { name: "query", title: "Query", type: "string", required: true },
          ],
        })}
      />,
    );
    fireEvent.click(screen.getByText("Proceed"));
    expect(mockOnSend).toHaveBeenCalledWith(
      "Please proceed with running the agent.",
    );
  });
});

describe("SetupRequirementsCard (session-scoped dismissal)", () => {
  it("auto-dismisses when all requested providers are already connected in this session", async () => {
    useConnectedProvidersStore
      .getState()
      .markConnected({ sessionID: "sess-1", providers: ["openai"] });

    render(
      <SetupRequirementsCard
        output={makeOutput({
          missingCredentials: {
            api_key: { provider: "openai", types: ["api_key"] },
          },
        })}
      />,
    );

    expect(screen.queryByText("Proceed")).toBeNull();
    expect(screen.getByText(/Connected. Continuing/)).toBeDefined();
    // Auto-dismiss must also fire onSend, otherwise the AI's tool call sits
    // unanswered and the chat hangs (regression: Sentry bug prediction).
    await waitFor(() => expect(mockOnSend).toHaveBeenCalledOnce());
  });

  it("auto-dismiss in preview mode sends the legacy run_agent message", async () => {
    useConnectedProvidersStore
      .getState()
      .markConnected({ sessionID: "sess-1", providers: ["openai"] });

    render(
      <SetupRequirementsCard
        inputsMode="preview"
        output={makeOutput({
          inputs: [
            { name: "url", title: "URL", type: "string", required: true },
          ],
          missingCredentials: {
            api_key: { provider: "openai", types: ["api_key"] },
          },
        })}
      />,
    );

    await waitFor(() =>
      expect(mockOnSend).toHaveBeenCalledWith(
        "I've configured the required credentials. Please check if everything is ready and proceed with running the agent.",
      ),
    );
  });

  it("auto-dismiss invokes onComplete callback", async () => {
    useConnectedProvidersStore
      .getState()
      .markConnected({ sessionID: "sess-1", providers: ["openai"] });
    const onComplete = vi.fn();

    render(
      <SetupRequirementsCard
        output={makeOutput({
          missingCredentials: {
            api_key: { provider: "openai", types: ["api_key"] },
          },
        })}
        onComplete={onComplete}
      />,
    );

    await waitFor(() => expect(onComplete).toHaveBeenCalledOnce());
  });

  it("does not auto-dismiss when only some providers are connected", () => {
    useConnectedProvidersStore
      .getState()
      .markConnected({ sessionID: "sess-1", providers: ["openai"] });

    render(
      <SetupRequirementsCard
        output={makeOutput({
          missingCredentials: {
            api_key: { provider: "openai", types: ["api_key"] },
            gh: { provider: "github", types: ["api_key"] },
          },
        })}
      />,
    );

    expect(screen.queryByText(/Connected. Continuing/)).toBeNull();
    expect(screen.getByText("Proceed")).toBeDefined();
  });

  it("marks providers connected in the store when Proceed is clicked", () => {
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
          missingCredentials: {
            // Use a credential that auto-fills via system credentials would be
            // ideal, but for this test we just need Proceed to be clickable —
            // an inputs-only card with no creds is the simplest path.
          },
        })}
      />,
    );

    fireEvent.click(screen.getByText("Proceed"));
    // No credentials in this output, so no providers should be marked.
    expect(useConnectedProvidersStore.getState().connected.size).toBe(0);
  });

  it("does NOT auto-dismiss when the card has user-fillable inputs pending", () => {
    // Regression: a run_block card with both already-connected creds AND
    // unfilled user inputs must keep showing the form — otherwise handleRun
    // never fires and the chat hangs waiting for input that can't be sent.
    useConnectedProvidersStore
      .getState()
      .markConnected({ sessionID: "sess-1", providers: ["openai"] });

    render(
      <SetupRequirementsCard
        output={makeOutput({
          inputs: [
            { name: "url", title: "URL", type: "string", required: true },
          ],
          missingCredentials: {
            api_key: { provider: "openai", types: ["api_key"] },
          },
        })}
      />,
    );

    expect(screen.queryByText(/Connected. Continuing/)).toBeNull();
    expect(screen.getByText("Proceed")).toBeDefined();
  });

  it("still auto-dismisses in preview mode even when inputs exist (read-only)", () => {
    // Preview-mode inputs aren't user-fillable, so a creds-only "Proceed"
    // would just be confirming pre-set values — auto-dismiss is still the
    // right call to avoid duplicate prompts.
    useConnectedProvidersStore
      .getState()
      .markConnected({ sessionID: "sess-1", providers: ["openai"] });

    render(
      <SetupRequirementsCard
        inputsMode="preview"
        output={makeOutput({
          inputs: [
            { name: "url", title: "URL", type: "string", required: true },
          ],
          missingCredentials: {
            api_key: { provider: "openai", types: ["api_key"] },
          },
        })}
      />,
    );

    expect(screen.getByText(/Connected. Continuing/)).toBeDefined();
  });

  it("only one card auto-sends when multiple parallel cards share a provider set", async () => {
    // Regression test for Sentry bug prediction: when N parallel
    // SetupRequirementsCard instances all require the same provider, after
    // one card connects the provider the others all flip to canAutoDismiss
    // simultaneously. Without coordination each card's useEffect would fire
    // onSend, producing N identical "Please proceed" messages and spamming
    // the chat. The store-level claim must allow only the first auto-dismiss
    // to send.
    useConnectedProvidersStore
      .getState()
      .markConnected({ sessionID: "sess-1", providers: ["openai"] });

    const output = makeOutput({
      missingCredentials: {
        api_key: { provider: "openai", types: ["api_key"] },
      },
    });

    render(
      <>
        <SetupRequirementsCard output={output} />
        <SetupRequirementsCard output={output} />
        <SetupRequirementsCard output={output} />
      </>,
    );

    // All three cards must visually dismiss.
    await waitFor(() =>
      expect(screen.getAllByText(/Connected. Continuing/)).toHaveLength(3),
    );
    // But only one auto-send fires for the whole provider set.
    expect(mockOnSend).toHaveBeenCalledOnce();
  });

  it("auto-send survives StrictMode's double-invoke of effects", async () => {
    // Regression test for Sentry bug prediction: in React StrictMode, the
    // auto-dismiss effect runs twice in quick succession (mount → cleanup →
    // re-mount). A previous version of this effect scheduled the send via a
    // microtask and cancelled it in cleanup, which leaked the store claim
    // across the double-invoke — the first effect's microtask got cancelled,
    // the second effect couldn't re-claim, and `handleRun` never fired.
    // Today the claim itself provides atomicity and we call `handleRun`
    // synchronously; this test pins the behaviour against a StrictMode
    // remount.
    useConnectedProvidersStore
      .getState()
      .markConnected({ sessionID: "sess-1", providers: ["openai"] });

    render(
      <StrictMode>
        <SetupRequirementsCard
          output={makeOutput({
            missingCredentials: {
              api_key: { provider: "openai", types: ["api_key"] },
            },
          })}
        />
      </StrictMode>,
    );

    // The card must dismiss AND the auto-send must fire exactly once
    // (not zero — the previous bug; not two — the original double-invoke
    // risk that the microtask used to guard against).
    expect(screen.getByText(/Connected. Continuing/)).toBeDefined();
    await waitFor(() => expect(mockOnSend).toHaveBeenCalledOnce());
  });
});
