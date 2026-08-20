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
      expect(request.onUseToken).toHaveBeenCalledWith("secret-token");
    });

    it("submits the manual token on Enter", () => {
      const request = mcpRequest({ showManualToken: true });
      renderCard({ mcp: [request] });

      const input = screen.getByLabelText("API token for Notion");
      fireEvent.change(input, { target: { value: "secret-token" } });
      fireEvent.keyDown(input, { key: "Enter" });

      expect(request.onUseToken).toHaveBeenCalledWith("secret-token");
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
