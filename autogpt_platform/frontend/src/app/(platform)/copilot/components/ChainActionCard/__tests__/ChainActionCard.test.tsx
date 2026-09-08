import {
  CredentialsProvidersContext,
  type CredentialsProvidersContextType,
} from "@/providers/agent-credentials/credentials-provider";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ChainActionCard } from "../ChainActionCard";
import type {
  ConnectorRequest,
  InputsRequest,
  McpConnectorRequest,
  QuestionRequest,
} from "../helpers";

const mockOpenModal = vi.fn();
vi.mock("../../../useCopilotModal", () => ({
  useCopilotModal: () => ({ openModal: mockOpenModal }),
}));

vi.mock("@/app/api/__generated__/endpoints/integrations/integrations", () => ({
  useGetV1ListProviders: () => ({
    data: [{ name: "github", description: "Connect your GitHub account" }],
  }),
}));

vi.mock(
  "@/components/contextual/CredentialsInput/components/ConnectCredentialDialog/ConnectCredentialDialog",
  () => ({
    ConnectCredentialDialog: ({
      open,
      provider,
    }: {
      open: boolean;
      provider: string;
    }) => (open ? <div data-testid={`connect-dialog-${provider}`} /> : null),
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
  mockOpenModal.mockReset();
});

function connectorRequest(
  overrides: Partial<ConnectorRequest> = {},
): ConnectorRequest {
  return {
    id: "connector-1",
    fields: [
      [
        "credentials",
        {
          credentials_provider: ["github"],
          credentials_types: ["api_key"],
        },
      ],
    ],
    selected: {},
    onChange: vi.fn(),
    onConnected: vi.fn(),
    ...overrides,
  };
}

function mcpRequest(
  overrides: Partial<McpConnectorRequest> = {},
): McpConnectorRequest {
  return {
    id: "mcp-1",
    service: "Notion",
    serverUrl: "https://mcp.notion.com/mcp",
    connected: false,
    loading: false,
    error: null,
    showManualToken: false,
    authScheme: "bearer",
    onConnect: vi.fn(),
    onUseToken: vi.fn(),
    ...overrides,
  };
}

function inputsRequest(overrides: Partial<InputsRequest> = {}): InputsRequest {
  return {
    id: "inputs-1",
    schema: { type: "object", properties: {} },
    values: {},
    onChange: vi.fn(),
    hasAdvanced: false,
    showAdvanced: false,
    onToggleAdvanced: vi.fn(),
    ...overrides,
  };
}

function questionRequest(
  overrides: Partial<QuestionRequest> = {},
): QuestionRequest {
  return {
    id: "questions-1",
    questions: [{ question: "Which region?", keyword: "region" }],
    answers: {},
    onAnswer: vi.fn(),
    onSkip: vi.fn(),
    ...overrides,
  };
}

function renderCard(
  overrides: Partial<React.ComponentProps<typeof ChainActionCard>> = {},
) {
  const onProceed = vi.fn();
  const utils = render(
    <ChainActionCard
      connectors={[]}
      mcp={[]}
      inputs={[]}
      questions={[]}
      manualProceed={false}
      isReady
      onProceed={onProceed}
      {...overrides}
    />,
  );
  return { onProceed, ...utils };
}

describe("ChainActionCard", () => {
  it("renders nothing when the chain has nothing to ask", () => {
    const { container } = renderCard();
    expect(container.firstChild).toBeNull();
  });

  it("renders nothing when every ask resolves to nothing actionable", () => {
    const { container } = renderCard({
      connectors: [
        connectorRequest({
          fields: [["credentials", { credentials_types: ["api_key"] }]],
        }),
      ],
      inputs: [inputsRequest({ schema: null, hasAdvanced: false })],
      questions: [questionRequest({ questions: [] })],
    });
    expect(container.firstChild).toBeNull();
  });

  describe("connectors table", () => {
    it("renders a row with the provider name and metadata description", () => {
      renderCard({ connectors: [connectorRequest()] });

      expect(screen.getByText("Plug in what this needs")).toBeDefined();
      expect(screen.getByText("GitHub")).toBeDefined();
      expect(screen.getByText("Connect your GitHub account")).toBeDefined();
    });

    it("opens the connect dialog when Connect is clicked", () => {
      renderCard({ connectors: [connectorRequest()] });

      expect(screen.queryByTestId("connect-dialog-github")).toBeNull();
      fireEvent.click(screen.getByText("Connect"));
      expect(screen.getByTestId("connect-dialog-github")).toBeDefined();
    });

    it("shows Connected instead of the Connect button once a credential is selected", () => {
      renderCard({
        connectors: [
          connectorRequest({
            selected: {
              credentials: {
                id: "cred-1",
                provider: "github",
                type: "api_key",
              },
            },
          }),
        ],
      });

      expect(screen.getByText("Connected")).toBeDefined();
      expect(screen.queryByText("Connect")).toBeNull();
    });

    it("keeps a selection while the providers context is still loading", async () => {
      const selected = {
        id: "saved-1",
        provider: "github",
        type: "api_key" as const,
      };
      const request = connectorRequest({ selected: { credentials: selected } });

      render(
        // `null` is the provider context's "still loading" sentinel, so every
        // credential lookup misses — clearing then drops a good selection.
        <CredentialsProvidersContext.Provider value={null}>
          <ChainActionCard
            connectors={[request]}
            mcp={[]}
            inputs={[]}
            questions={[]}
            manualProceed={false}
            isReady
            onProceed={vi.fn()}
          />
        </CredentialsProvidersContext.Provider>,
      );

      await screen.findByText("GitHub");
      expect(request.onChange).not.toHaveBeenCalled();
    });

    it("clears the selection once loading reveals no matching credential", async () => {
      const selected = {
        id: "saved-1",
        provider: "github",
        type: "api_key" as const,
      };
      const request = connectorRequest({ selected: { credentials: selected } });
      const loadedWithNoMatch = {
        github: { savedCredentials: [] },
      } as unknown as CredentialsProvidersContextType;

      const { rerender } = render(
        <CredentialsProvidersContext.Provider value={null}>
          <ChainActionCard
            connectors={[request]}
            mcp={[]}
            inputs={[]}
            questions={[]}
            manualProceed={false}
            isReady
            onProceed={vi.fn()}
          />
        </CredentialsProvidersContext.Provider>,
      );
      expect(request.onChange).not.toHaveBeenCalled();

      rerender(
        <CredentialsProvidersContext.Provider value={loadedWithNoMatch}>
          <ChainActionCard
            connectors={[request]}
            mcp={[]}
            inputs={[]}
            questions={[]}
            manualProceed={false}
            isReady
            onProceed={vi.fn()}
          />
        </CredentialsProvidersContext.Provider>,
      );

      // Neither the credential nor the selection changed across this
      // transition, so only `allProviders` can re-run the effect.
      await waitFor(() =>
        expect(request.onChange).toHaveBeenCalledWith("credentials", undefined),
      );
    });

    it("auto-selects a saved credential from the providers context", async () => {
      const request = connectorRequest();
      const providers = {
        github: {
          savedCredentials: [
            {
              id: "saved-1",
              provider: "github",
              type: "api_key",
              title: "My key",
            },
          ],
        },
      } as unknown as CredentialsProvidersContextType;

      render(
        <CredentialsProvidersContext.Provider value={providers}>
          <ChainActionCard
            connectors={[request]}
            mcp={[]}
            inputs={[]}
            questions={[]}
            manualProceed={false}
            isReady
            onProceed={vi.fn()}
          />
        </CredentialsProvidersContext.Provider>,
      );

      await waitFor(() =>
        expect(request.onChange).toHaveBeenCalledWith("credentials", {
          id: "saved-1",
          provider: "github",
          type: "api_key",
          title: "My key",
        }),
      );
    });

    it("opens the integrations modal from Browse all connectors", () => {
      renderCard({ connectors: [connectorRequest()] });

      fireEvent.click(screen.getByText("Browse all connectors"));
      expect(mockOpenModal).toHaveBeenCalledWith("integrations");
    });
  });

  describe("MCP rows", () => {
    it("dedupes rows that point at the same server URL", () => {
      renderCard({
        mcp: [
          mcpRequest({ id: "mcp-1" }),
          mcpRequest({ id: "mcp-2", service: "Notion again" }),
        ],
      });

      expect(screen.getByText("Notion")).toBeDefined();
      expect(screen.queryByText("Notion again")).toBeNull();
    });

    it("shows the server host and calls onConnect from the Connect button", () => {
      const request = mcpRequest();
      renderCard({ mcp: [request] });

      expect(screen.getByText("mcp.notion.com")).toBeDefined();
      fireEvent.click(screen.getByText("Connect"));
      expect(request.onConnect).toHaveBeenCalledOnce();
    });

    it("omits the host line when the server URL is not parseable", () => {
      renderCard({ mcp: [mcpRequest({ serverUrl: "not-a-url" })] });

      expect(screen.getByText("Notion")).toBeDefined();
      expect(screen.queryByText("not-a-url")).toBeNull();
    });

    it("disables the button and shows Connecting… while loading", () => {
      renderCard({ mcp: [mcpRequest({ loading: true })] });

      const button = screen.getByText("Connecting…").closest("button");
      expect(button?.disabled).toBe(true);
    });

    it("shows Connected once the server is connected", () => {
      renderCard({ mcp: [mcpRequest({ connected: true })] });

      expect(screen.getByText("Connected")).toBeDefined();
      expect(screen.queryByText("Connect")).toBeNull();
    });

    it("shows the error message while disconnected", () => {
      renderCard({ mcp: [mcpRequest({ error: "OAuth failed" })] });
      expect(screen.getByText("OAuth failed")).toBeDefined();
    });

    it("submits a trimmed manual token via the Use Token button", () => {
      const request = mcpRequest({ showManualToken: true });
      renderCard({ mcp: [request] });

      const useToken = screen.getByText("Use Token").closest("button");
      expect(useToken?.disabled).toBe(true);

      fireEvent.change(screen.getByLabelText("API token for Notion"), {
        target: { value: "  secret-token  " },
      });
      expect(useToken?.disabled).toBe(false);

      fireEvent.click(useToken!);
      expect(request.onUseToken).toHaveBeenCalledWith("Bearer secret-token");
    });

    it("submits the manual token on Enter", () => {
      const request = mcpRequest({ showManualToken: true });
      renderCard({ mcp: [request] });

      const input = screen.getByLabelText("API token for Notion");
      fireEvent.change(input, { target: { value: "secret-token" } });
      fireEvent.keyDown(input, { key: "Enter" });

      expect(request.onUseToken).toHaveBeenCalledWith("Bearer secret-token");
    });

    it("prefixes a Basic credential selected in the chain row", () => {
      const request = mcpRequest({ showManualToken: true });
      renderCard({ mcp: [request] });

      fireEvent.change(
        screen.getByLabelText("Authentication type for Notion"),
        { target: { value: "basic" } },
      );
      fireEvent.change(
        screen.getByLabelText("Basic authentication token for Notion"),
        { target: { value: "  cGstbGYtYWJjZA==  " } },
      );
      fireEvent.click(screen.getByText("Use Token"));

      expect(request.onUseToken).toHaveBeenCalledWith("Basic cGstbGYtYWJjZA==");
    });

    it("refuses an unencoded user:password before calling onUseToken", () => {
      // The other three manual-credential surfaces validate before storing.
      // Without it this one sent the raw pair, and the user got the backend's
      // 422 instead of the message that names the Base64 step.
      const request = mcpRequest({ showManualToken: true });
      renderCard({ mcp: [request] });

      fireEvent.change(
        screen.getByLabelText("Authentication type for Notion"),
        { target: { value: "basic" } },
      );
      fireEvent.change(
        screen.getByLabelText("Basic authentication token for Notion"),
        { target: { value: "pk-lf-abc:sk-lf-xyz" } },
      );
      fireEvent.click(screen.getByText("Use Token"));

      expect(request.onUseToken).not.toHaveBeenCalled();
      expect(screen.getByRole("alert").textContent).toMatch(
        /unencoded user:password/,
      );
    });

    it("points the credential input at both its hint and its error", () => {
      const request = mcpRequest({ showManualToken: true });
      renderCard({ mcp: [request] });

      const input = screen.getByLabelText("API token for Notion");
      const hintId = input.getAttribute("aria-describedby");
      expect(hintId).toBeTruthy();
      // The hint is the only place the Base64 step is explained, and nothing
      // carried the id the input pointed at.
      expect(document.getElementById(hintId as string)?.textContent).toMatch(
        /Bearer authentication/,
      );
    });

    it("uses the selector when it differs from a pasted prefix", () => {
      const request = mcpRequest({ showManualToken: true });
      renderCard({ mcp: [request] });

      fireEvent.change(screen.getByLabelText("API token for Notion"), {
        target: { value: "Basic encoded-value" },
      });
      fireEvent.change(
        screen.getByLabelText("Authentication type for Notion"),
        { target: { value: "bearer" } },
      );
      fireEvent.click(screen.getByText("Use Token"));

      expect(request.onUseToken).toHaveBeenCalledWith("Bearer encoded-value");
    });
  });

  describe("inputs cards", () => {
    it("titles the card by the formatted block name", () => {
      renderCard({
        inputs: [inputsRequest({ title: "SendDiscordMessageBlock" })],
      });

      expect(screen.getByText("Send Discord Message")).toBeDefined();
      expect(screen.getByText("Fill in the details")).toBeDefined();
    });

    it("falls back to a generic title when the request has none", () => {
      renderCard({ inputs: [inputsRequest()] });
      expect(screen.getByText("Fill in the details")).toBeDefined();
    });

    it("forwards form changes to the request", () => {
      const request = inputsRequest();
      renderCard({ inputs: [request] });

      fireEvent.click(screen.getByTestId("form-change"));
      expect(request.onChange).toHaveBeenCalledWith({
        url: "https://test.com",
      });
    });

    it("toggles advanced fields through the request callback", () => {
      const request = inputsRequest({ hasAdvanced: true });
      renderCard({ inputs: [request] });

      fireEvent.click(screen.getByText("Show advanced fields"));
      expect(request.onToggleAdvanced).toHaveBeenCalledOnce();
    });

    it("labels the toggle Hide advanced fields when advanced fields are shown", () => {
      renderCard({
        inputs: [inputsRequest({ hasAdvanced: true, showAdvanced: true })],
      });
      expect(screen.getByText("Hide advanced fields")).toBeDefined();
    });

    it("renders a single Proceed for an inputs-only stack gated on isReady", () => {
      const { onProceed, rerender } = renderCard({
        inputs: [inputsRequest()],
        isReady: false,
      });

      const proceed = screen.getByText("Proceed").closest("button");
      expect(proceed?.disabled).toBe(true);

      rerender(
        <ChainActionCard
          connectors={[]}
          mcp={[]}
          inputs={[inputsRequest()]}
          questions={[]}
          manualProceed={false}
          isReady
          onProceed={onProceed}
        />,
      );
      fireEvent.click(screen.getByText("Proceed"));
      expect(onProceed).toHaveBeenCalledOnce();
    });

    it("does not render the lone Proceed when a questions card carries the footer", () => {
      renderCard({
        inputs: [inputsRequest()],
        questions: [questionRequest()],
      });
      expect(screen.queryByText("Proceed")).toBeNull();
    });
  });

  describe("questions stepper", () => {
    const twoQuestions = [
      { question: "Which region?", keyword: "region" },
      { question: "Which format?", keyword: "format", example: "CSV" },
    ];

    it("gates the action button on the current answer", () => {
      renderCard({ questions: [questionRequest({ questions: twoQuestions })] });

      expect(screen.getByText("Answer a few questions")).toBeDefined();
      expect(screen.getByText("Which region?")).toBeDefined();

      const nextButtons = screen.getAllByRole("button", {
        name: "Next question",
      }) as HTMLButtonElement[];
      // The round action is answer-gated; the pager chevron is not.
      expect(nextButtons.filter((button) => button.disabled)).toHaveLength(1);
    });

    it("advances to the next question once answered", () => {
      renderCard({
        questions: [
          questionRequest({
            questions: twoQuestions,
            answers: { region: "Europe" },
          }),
        ],
      });

      const [action] = (
        screen.getAllByRole("button", {
          name: "Next question",
        }) as HTMLButtonElement[]
      ).filter((button) => !button.disabled);
      fireEvent.click(action);

      expect(screen.getByText("Which format?")).toBeDefined();
      expect(screen.getByPlaceholderText("e.g. CSV")).toBeDefined();
    });

    it("navigates with the dots and the back chevron", () => {
      renderCard({
        questions: [
          questionRequest({
            questions: twoQuestions,
            answers: { region: "Europe", format: "CSV" },
          }),
        ],
      });

      const back = screen.getByRole("button", {
        name: "Previous question",
      }) as HTMLButtonElement;
      expect(back.disabled).toBe(true);

      fireEvent.click(screen.getByRole("button", { name: "Go to question 2" }));
      expect(screen.getByText("Which format?")).toBeDefined();

      fireEvent.click(
        screen.getByRole("button", { name: "Previous question" }),
      );
      expect(screen.getByText("Which region?")).toBeDefined();
    });

    it("skips every request from the Skip button", () => {
      const first = questionRequest({ id: "q-1" });
      const second = questionRequest({
        id: "q-2",
        questions: [{ question: "Which format?", keyword: "format" }],
      });
      renderCard({ questions: [first, second] });

      fireEvent.click(screen.getByText("Skip"));
      expect(first.onSkip).toHaveBeenCalledOnce();
      expect(second.onSkip).toHaveBeenCalledOnce();
    });

    it("forwards typing to the request's onAnswer", () => {
      const request = questionRequest();
      renderCard({ questions: [request] });

      fireEvent.change(screen.getByPlaceholderText("Type your answer"), {
        target: { value: "Europe" },
      });
      expect(request.onAnswer).toHaveBeenCalledWith("region", "Europe");
    });

    it("disables the send button until every answer is filled", () => {
      renderCard({ questions: [questionRequest()] });

      const send = screen.getByRole("button", {
        name: "Add answers to message",
      }) as HTMLButtonElement;
      expect(send.disabled).toBe(true);
    });

    it("drafts the answers on the last step when every answer is filled", () => {
      const { onProceed } = renderCard({
        questions: [questionRequest({ answers: { region: "Europe" } })],
      });

      fireEvent.click(
        screen.getByRole("button", { name: "Add answers to message" }),
      );
      expect(onProceed).toHaveBeenCalledOnce();
    });

    it("submits the last step on Enter", () => {
      const { onProceed } = renderCard({
        questions: [questionRequest({ answers: { region: "Europe" } })],
      });

      fireEvent.keyDown(screen.getByDisplayValue("Europe"), { key: "Enter" });
      expect(onProceed).toHaveBeenCalledOnce();
    });

    it("renders options as choices and selects one on click", () => {
      const request = questionRequest({
        questions: [
          {
            question: "Which region?",
            keyword: "region",
            options: ["Europe", "Americas"],
          },
        ],
      });
      renderCard({ questions: [request] });

      expect(screen.queryByPlaceholderText("Type your answer")).toBeNull();
      fireEvent.click(screen.getByRole("radio", { name: "Europe" }));
      expect(request.onAnswer).toHaveBeenCalledWith("region", "Europe");
    });

    it("marks the option matching the answer as selected", () => {
      renderCard({
        questions: [
          questionRequest({
            questions: [
              {
                question: "Which region?",
                keyword: "region",
                options: ["Europe", "Americas"],
              },
            ],
            answers: { region: "Europe" },
          }),
        ],
      });

      expect(
        screen
          .getByRole("radio", { name: "Europe" })
          .getAttribute("aria-checked"),
      ).toBe("true");
      expect(
        screen
          .getByRole("radio", { name: "Americas" })
          .getAttribute("aria-checked"),
      ).toBe("false");
    });

    it("swaps to free text via Type something and clears a picked option", () => {
      const request = questionRequest({
        questions: [
          {
            question: "Which region?",
            keyword: "region",
            options: ["Europe", "Americas"],
          },
        ],
        answers: { region: "Europe" },
      });
      renderCard({ questions: [request] });

      fireEvent.click(screen.getByText("Type something…"));
      expect(request.onAnswer).toHaveBeenCalledWith("region", "");
      expect(screen.getByPlaceholderText("Type your answer")).toBeDefined();

      fireEvent.click(screen.getByText("Choose from options instead"));
      expect(screen.getByRole("radio", { name: "Europe" })).toBeDefined();
    });

    it("opens in free text when the answer matches no option", () => {
      renderCard({
        questions: [
          questionRequest({
            questions: [
              {
                question: "Which region?",
                keyword: "region",
                options: ["Europe", "Americas"],
              },
            ],
            answers: { region: "Antarctica" },
          }),
        ],
      });

      expect(screen.getByDisplayValue("Antarctica")).toBeDefined();
    });

    it("clears a custom answer when going back to the options", () => {
      const request = questionRequest({
        questions: [
          {
            question: "Which region?",
            keyword: "region",
            options: ["Europe", "Americas"],
          },
        ],
        answers: { region: "Antarctica" },
      });
      renderCard({ questions: [request] });

      fireEvent.click(screen.getByText("Choose from options instead"));
      expect(request.onAnswer).toHaveBeenCalledWith("region", "");
    });

    it("names the option group after the question", () => {
      renderCard({
        questions: [
          questionRequest({
            questions: [
              {
                question: "Which region?",
                keyword: "region",
                options: ["Europe", "Americas"],
              },
            ],
          }),
        ],
      });

      expect(
        screen.getByRole("radiogroup", { name: "Which region?" }),
      ).toBeDefined();
    });

    it("moves and selects between options with the arrow keys", () => {
      const request = questionRequest({
        questions: [
          {
            question: "Which region?",
            keyword: "region",
            options: ["Europe", "Americas"],
          },
        ],
        answers: { region: "Europe" },
      });
      renderCard({ questions: [request] });

      fireEvent.keyDown(screen.getByRole("radio", { name: "Europe" }), {
        key: "ArrowDown",
      });
      expect(request.onAnswer).toHaveBeenCalledWith("region", "Americas");
    });

    it("keeps only the active option in the tab order", () => {
      renderCard({
        questions: [
          questionRequest({
            questions: [
              {
                question: "Which region?",
                keyword: "region",
                options: ["Europe", "Americas"],
              },
            ],
            answers: { region: "Americas" },
          }),
        ],
      });

      expect(
        screen
          .getByRole("radio", { name: "Americas" })
          .getAttribute("tabindex"),
      ).toBe("0");
      expect(
        screen.getByRole("radio", { name: "Europe" }).getAttribute("tabindex"),
      ).toBe("-1");
    });

    it("does not steal focus when the first question opens in free text", () => {
      renderCard({
        questions: [
          questionRequest({
            questions: [
              {
                question: "Which region?",
                keyword: "region",
                options: ["Europe", "Americas"],
              },
            ],
            answers: { region: "Antarctica" },
          }),
        ],
      });

      expect(document.activeElement).toBe(document.body);
    });

    it("focuses the free-text box when the user asks to type", () => {
      renderCard({
        questions: [
          questionRequest({
            questions: [
              {
                question: "Which region?",
                keyword: "region",
                options: ["Europe", "Americas"],
              },
            ],
          }),
        ],
      });

      fireEvent.click(screen.getByText("Type something…"));
      expect(document.activeElement).toBe(
        screen.getByPlaceholderText("Type your answer"),
      );
    });

    it("focuses the active option when the pager reaches a question", () => {
      renderCard({
        questions: [
          questionRequest({
            questions: [
              { question: "Which region?", keyword: "region" },
              {
                question: "Which channel?",
                keyword: "channel",
                options: ["Email", "Slack"],
              },
            ],
            answers: { region: "Europe" },
          }),
        ],
      });

      fireEvent.click(screen.getByRole("button", { name: "Go to question 2" }));
      expect(document.activeElement).toBe(
        screen.getByRole("radio", { name: "Email" }),
      );
    });

    it("does not leak the free-text toggle between same-keyword requests", () => {
      renderCard({
        questions: [
          questionRequest({
            id: "questions-1",
            questions: [
              {
                question: "Which region?",
                keyword: "region",
                options: ["Europe", "Americas"],
              },
            ],
            answers: { region: "Europe" },
          }),
          questionRequest({
            id: "questions-2",
            questions: [
              {
                question: "Which region?",
                keyword: "region",
                options: ["Notion", "Drive"],
              },
            ],
            answers: {},
          }),
        ],
      });

      fireEvent.click(screen.getByText("Type something…"));
      expect(screen.getByPlaceholderText("Type your answer")).toBeDefined();

      fireEvent.click(screen.getByRole("button", { name: "Go to question 2" }));
      expect(screen.getByRole("radio", { name: "Notion" })).toBeDefined();
    });

    it("moves focus onto the option the arrow keys select", () => {
      renderCard({
        questions: [
          questionRequest({
            questions: [
              {
                question: "Which region?",
                keyword: "region",
                options: ["Europe", "Americas"],
              },
            ],
            answers: { region: "Europe" },
          }),
        ],
      });

      fireEvent.keyDown(screen.getByRole("radio", { name: "Europe" }), {
        key: "ArrowDown",
      });
      expect(document.activeElement).toBe(
        screen.getByRole("radio", { name: "Americas" }),
      );
    });

    it("selects backwards with ArrowUp and ArrowLeft", () => {
      const request = questionRequest({
        questions: [
          {
            question: "Which region?",
            keyword: "region",
            options: ["Europe", "Americas", "Asia"],
          },
        ],
        answers: { region: "Asia" },
      });
      renderCard({ questions: [request] });

      fireEvent.keyDown(screen.getByRole("radio", { name: "Asia" }), {
        key: "ArrowUp",
      });
      expect(request.onAnswer).toHaveBeenCalledWith("region", "Americas");

      fireEvent.keyDown(screen.getByRole("radio", { name: "Asia" }), {
        key: "ArrowLeft",
      });
      expect(request.onAnswer).toHaveBeenLastCalledWith("region", "Americas");
    });

    it("wraps around at both ends of the option list", () => {
      // Without the modulo, ArrowUp on the first option reads options[-1] and
      // answers the question with undefined.
      const first = questionRequest({
        questions: [
          {
            question: "Which region?",
            keyword: "region",
            options: ["Europe", "Americas", "Asia"],
          },
        ],
        answers: { region: "Europe" },
      });
      renderCard({ questions: [first] });
      fireEvent.keyDown(screen.getByRole("radio", { name: "Europe" }), {
        key: "ArrowUp",
      });
      expect(first.onAnswer).toHaveBeenCalledWith("region", "Asia");

      cleanup();

      const last = questionRequest({
        questions: [
          {
            question: "Which region?",
            keyword: "region",
            options: ["Europe", "Americas", "Asia"],
          },
        ],
        answers: { region: "Asia" },
      });
      renderCard({ questions: [last] });
      fireEvent.keyDown(screen.getByRole("radio", { name: "Asia" }), {
        key: "ArrowDown",
      });
      expect(last.onAnswer).toHaveBeenCalledWith("region", "Europe");
    });

    it("submits from the option list on Enter once an option is selected", () => {
      const request = questionRequest({
        questions: [
          {
            question: "Which region?",
            keyword: "region",
            options: ["Europe", "Americas"],
          },
        ],
        answers: { region: "Europe" },
      });
      const { onProceed } = renderCard({ questions: [request] });

      fireEvent.keyDown(screen.getByRole("radio", { name: "Europe" }), {
        key: "Enter",
      });
      expect(onProceed).toHaveBeenCalledOnce();
    });

    it("selects rather than submits on Enter when nothing is chosen yet", () => {
      const request = questionRequest({
        questions: [
          {
            question: "Which region?",
            keyword: "region",
            options: ["Europe", "Americas"],
          },
        ],
      });
      const { onProceed } = renderCard({ questions: [request] });

      fireEvent.keyDown(screen.getByRole("radio", { name: "Europe" }), {
        key: "Enter",
      });
      expect(request.onAnswer).toHaveBeenCalledWith("region", "Europe");
      expect(onProceed).not.toHaveBeenCalled();
    });

    it("matches the selected option against the trimmed answer", () => {
      renderCard({
        questions: [
          questionRequest({
            questions: [
              {
                question: "Which region?",
                keyword: "region",
                options: ["Europe", "Americas"],
              },
            ],
            answers: { region: "  Europe  " },
          }),
        ],
      });

      expect(
        screen
          .getByRole("radio", { name: "Europe" })
          .getAttribute("aria-checked"),
      ).toBe("true");
    });

    it("does not replay the declined options as the free-text placeholder", () => {
      // The backend sets example to the joined options, so echoing it back to
      // someone who just left the option list would be pure noise.
      renderCard({
        questions: [
          questionRequest({
            questions: [
              {
                question: "Which region?",
                keyword: "region",
                example: "Europe, Americas",
                options: ["Europe", "Americas"],
              },
            ],
          }),
        ],
      });

      fireEvent.click(screen.getByText("Type something…"));
      expect(screen.getByPlaceholderText("Type your answer")).toBeDefined();
      expect(screen.queryByPlaceholderText("e.g. Europe, Americas")).toBeNull();
    });

    it("gates the send button on an option actually being picked", () => {
      const request = questionRequest({
        questions: [
          {
            question: "Which region?",
            keyword: "region",
            options: ["Europe", "Americas"],
          },
        ],
      });
      const { onProceed, rerender } = renderCard({ questions: [request] });

      expect(
        (
          screen.getByRole("button", {
            name: "Add answers to message",
          }) as HTMLButtonElement
        ).disabled,
      ).toBe(true);

      fireEvent.click(screen.getByRole("radio", { name: "Europe" }));
      expect(request.onAnswer).toHaveBeenCalledWith("region", "Europe");

      rerender(
        <ChainActionCard
          connectors={[]}
          mcp={[]}
          inputs={[]}
          questions={[{ ...request, answers: { region: "Europe" } }]}
          manualProceed={false}
          isReady
          onProceed={onProceed}
        />,
      );

      const send = screen.getByRole("button", {
        name: "Add answers to message",
      }) as HTMLButtonElement;
      expect(send.disabled).toBe(false);
      fireEvent.click(send);
      expect(onProceed).toHaveBeenCalledOnce();
    });

    it("keeps two same-keyword questions on their own cards", () => {
      // Keywords are unique only within a request, so the pager keys on
      // position — sharing an id would stop the field remounting, leaving the
      // second question's options blank while the first one's answer submits.
      renderCard({
        questions: [
          questionRequest({
            questions: [
              {
                question: "Which source channel?",
                keyword: "channel",
                options: ["Email", "Slack"],
              },
              {
                question: "Which destination channel?",
                keyword: "channel",
                options: ["Notion", "Drive"],
              },
            ],
            answers: { channel: "Email" },
          }),
        ],
      });

      expect(screen.getByRole("radio", { name: "Email" })).toBeDefined();
      fireEvent.click(screen.getByRole("button", { name: "Go to question 2" }));

      // Remounted: "Email" matches neither option here, so the field opens in
      // free text with the stray value visible instead of silently hiding it.
      expect(screen.getByDisplayValue("Email")).toBeDefined();
      expect(screen.queryByRole("radio", { name: "Notion" })).toBeNull();
    });

    it("keeps the questions card sendable when an unready sibling exists", () => {
      // An unconnected MCP row must not freeze the questions footer.
      const { onProceed } = renderCard({
        mcp: [mcpRequest()],
        questions: [questionRequest({ answers: { region: "Europe" } })],
        isReady: false,
      });

      fireEvent.click(
        screen.getByRole("button", { name: "Add answers to message" }),
      );
      expect(onProceed).toHaveBeenCalledOnce();
    });
  });
});
