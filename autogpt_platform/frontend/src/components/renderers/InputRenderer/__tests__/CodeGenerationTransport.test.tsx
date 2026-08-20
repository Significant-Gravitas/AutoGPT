import { BlockUIType } from "@/app/(platform)/build/components/types";
import { FormCreator } from "@/app/(platform)/build/components/FlowEditor/nodes/FormCreator";
import type { CustomNode } from "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/CustomNode";
import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";
import { useHistoryStore } from "@/app/(platform)/build/stores/historyStore";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import {
  CredentialsProvidersContext,
  type CredentialsProviderData,
} from "@/providers/agent-credentials/credentials-provider";
import { render } from "@/tests/integrations/test-utils";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import type { RJSFSchema } from "@rjsf/utils";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/components/atoms/Select/Select", () => {
  function Select({
    id,
    value,
    onValueChange,
    options,
  }: {
    id: string;
    value?: string;
    onValueChange?: (value: string) => void;
    options: { value: string; label: string }[];
  }) {
    return (
      <select
        aria-label={id}
        value={value ?? ""}
        onChange={(event) => onValueChange?.(event.target.value)}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    );
  }

  return { Select };
});

const codeGenerationSchema = {
  type: "object",
  properties: {
    prompt: {
      advanced: false,
      title: "Prompt",
      type: "string",
    },
    system_prompt: {
      advanced: true,
      default: "You are Codex.",
      title: "System Prompt",
      type: "string",
    },
    transport: {
      advanced: false,
      default: "openai_api",
      enum: ["openai_api", "codex_app_server"],
      title: "Transport",
      type: "string",
    },
    model: {
      advanced: false,
      default: "gpt-5.3-codex",
      enum: ["gpt-5.3-codex", "gpt-5.1-codex"],
      title: "Codex Model",
      type: "string",
    },
    reasoning_effort: {
      advanced: true,
      default: "medium",
      enum: ["none", "low", "medium", "high", "xhigh"],
      title: "Reasoning Effort",
      type: "string",
    },
    credentials: {
      credentials_provider: ["openai", "codex"],
      credentials_types: ["api_key", "oauth2"],
      discriminator: "transport",
      discriminator_mapping: {
        openai_api: "openai",
        codex_app_server: "codex",
      },
      discriminator_type_mapping: {
        openai_api: ["api_key"],
        codex_app_server: ["oauth2"],
      },
      properties: {
        id: { type: "string" },
        provider: { enum: ["openai", "codex"], type: "string" },
        type: { enum: ["api_key", "oauth2"], type: "string" },
      },
      required: ["id", "provider", "type"],
      title: "Credentials",
      type: "object",
    },
  },
  required: ["prompt", "credentials"],
} as unknown as RJSFSchema;

const codexCredential = {
  id: "codex-oauth-1",
  provider: "codex",
  type: "oauth2" as const,
  title: "ChatGPT for Codex",
  username: "nick@ntindle.com",
  scopes: [],
};

function makeProvider(
  provider: string,
  providerName: string,
  savedCredentials: CredentialsProviderData["savedCredentials"],
): CredentialsProviderData {
  return {
    provider,
    providerName,
    savedCredentials,
    isSystemProvider: false,
    oAuthCallback: async () => codexCredential,
    mcpOAuthCallback: async () => codexCredential,
    createAPIKeyCredentials: async () => codexCredential,
    createUserPasswordCredentials: async () => codexCredential,
    createHostScopedCredentials: async () => codexCredential,
    deleteCredentials: async () => ({ deleted: true, revoked: null }),
  };
}

function createCodeGenerationNode(): CustomNode {
  return {
    id: "code-generation-node",
    type: "custom",
    position: { x: 0, y: 0 },
    data: {
      hardcodedValues: {
        prompt: "Write a hello-world function",
        transport: "openai_api",
      },
      title: "Code Generation",
      description: "Generate code with Codex",
      inputSchema: codeGenerationSchema,
      outputSchema: {},
      uiType: BlockUIType.STANDARD,
      block_id: "86a2a099-30df-47b4-b7e4-34ae5f83e0d5",
      costs: [],
      categories: [],
    },
  };
}

beforeEach(() => {
  useNodeStore.setState({
    nodes: [createCodeGenerationNode()],
    nodeAdvancedStates: {},
  });
  useEdgeStore.setState({ edges: [] });
  useHistoryStore.setState({ past: [], future: [] });
});

describe("Code Generation transport fields", () => {
  it("keeps Transport visible and selects the saved Codex OAuth credential", async () => {
    const providers = {
      openai: makeProvider("openai", "OpenAI", []),
      codex: makeProvider("codex", "Codex", [codexCredential]),
    };

    render(
      <CredentialsProvidersContext.Provider value={providers}>
        <FormCreator
          jsonSchema={codeGenerationSchema}
          nodeId="code-generation-node"
          uiType={BlockUIType.STANDARD}
          showHandles={false}
        />
      </CredentialsProvidersContext.Provider>,
    );

    expect(screen.getByText("Transport")).toBeDefined();
    expect(screen.queryByText("System Prompt")).toBeNull();
    expect(screen.queryByText("Reasoning Effort")).toBeNull();

    fireEvent.change(screen.getByLabelText("agpt_%_transport"), {
      target: { value: "1" },
    });

    expect(await screen.findByText("ChatGPT for Codex")).toBeDefined();
    expect(screen.getByText("OAuth")).toBeDefined();

    await waitFor(() => {
      expect(
        useNodeStore.getState().getHardCodedValues("code-generation-node"),
      ).toMatchObject({
        transport: "codex_app_server",
        credentials: {
          id: "codex-oauth-1",
          provider: "codex",
          type: "oauth2",
          title: "ChatGPT for Codex",
        },
      });
    });
  });
});

function renderTransport(
  providers: Record<string, CredentialsProviderData> | null,
) {
  return render(
    <CredentialsProvidersContext.Provider value={providers}>
      <FormCreator
        jsonSchema={codeGenerationSchema}
        nodeId="code-generation-node"
        uiType={BlockUIType.STANDARD}
        showHandles={false}
      />
    </CredentialsProvidersContext.Provider>,
  );
}

function transportOptionLabels() {
  const select = screen.getByLabelText("agpt_%_transport");
  return Array.from(select.querySelectorAll("option")).map(
    (option) => option.textContent,
  );
}

describe("Transport options gated by provider entitlement", () => {
  it("hides the codex transport when the provider is not on the account", () => {
    renderTransport({ openai: makeProvider("openai", "OpenAI", []) });

    expect(transportOptionLabels()).toEqual(["openai_api"]);
  });

  it("offers the codex transport when the provider is on the account", () => {
    renderTransport({
      openai: makeProvider("openai", "OpenAI", []),
      codex: makeProvider("codex", "Codex", [codexCredential]),
    });

    expect(transportOptionLabels()).toEqual(["openai_api", "codex_app_server"]);
  });

  it("leaves options alone while the provider map is still loading", () => {
    renderTransport(null);

    expect(transportOptionLabels()).toEqual(["openai_api", "codex_app_server"]);
  });

  it("keeps a saved codex transport selectable after entitlement is lost", () => {
    useNodeStore.setState({
      nodes: [
        {
          ...createCodeGenerationNode(),
          data: {
            ...createCodeGenerationNode().data,
            hardcodedValues: {
              prompt: "Write a hello-world function",
              transport: "codex_app_server",
            },
          },
        },
      ],
      nodeAdvancedStates: {},
    });

    renderTransport({ openai: makeProvider("openai", "OpenAI", []) });

    expect(transportOptionLabels()).toEqual(["openai_api", "codex_app_server"]);
  });

  it("leaves the enum intact when every option would be filtered out", () => {
    // An empty dropdown is worse than an unusable option. Needs a node with
    // no saved transport and a schema with no default, otherwise those two
    // escape hatches keep an option and the guard never fires.
    const noDefault = JSON.parse(
      JSON.stringify(codeGenerationSchema),
    ) as Record<string, any>;
    delete noDefault.properties.transport.default;

    useNodeStore.setState({
      nodes: [
        {
          ...createCodeGenerationNode(),
          data: {
            ...createCodeGenerationNode().data,
            hardcodedValues: { prompt: "hi" },
            inputSchema: noDefault,
          },
        },
      ],
      nodeAdvancedStates: {},
    });

    render(
      <CredentialsProvidersContext.Provider value={{}}>
        <FormCreator
          jsonSchema={noDefault as unknown as RJSFSchema}
          nodeId="code-generation-node"
          uiType={BlockUIType.STANDARD}
          showHandles={false}
        />
      </CredentialsProvidersContext.Provider>,
    );

    expect(transportOptionLabels()).toEqual(["openai_api", "codex_app_server"]);
  });

  it("keeps the schema default even when its provider is gated", () => {
    // RJSF falls back to the schema default when the node has no saved value;
    // filtering it out would leave a select whose selection is not an option.
    useNodeStore.setState({
      nodes: [
        {
          ...createCodeGenerationNode(),
          data: {
            ...createCodeGenerationNode().data,
            hardcodedValues: { prompt: "hi" },
            inputSchema: {
              ...(codeGenerationSchema as Record<string, any>),
              properties: {
                ...(codeGenerationSchema as Record<string, any>).properties,
                transport: {
                  ...(codeGenerationSchema as Record<string, any>).properties
                    .transport,
                  default: "codex_app_server",
                },
              },
            },
          },
        },
      ],
      nodeAdvancedStates: {},
    });

    render(
      <CredentialsProvidersContext.Provider
        value={{ openai: makeProvider("openai", "OpenAI", []) }}
      >
        <FormCreator
          jsonSchema={
            {
              ...(codeGenerationSchema as Record<string, any>),
              properties: {
                ...(codeGenerationSchema as Record<string, any>).properties,
                transport: {
                  ...(codeGenerationSchema as Record<string, any>).properties
                    .transport,
                  default: "codex_app_server",
                },
              },
            } as unknown as RJSFSchema
          }
          nodeId="code-generation-node"
          uiType={BlockUIType.STANDARD}
          showHandles={false}
        />
      </CredentialsProvidersContext.Provider>,
    );

    expect(transportOptionLabels()).toContain("codex_app_server");
  });

  it("does not touch the model dropdown, which no credential discriminates on", () => {
    renderTransport({ openai: makeProvider("openai", "OpenAI", []) });

    const model = screen.getByLabelText("agpt_%_model");
    expect(
      Array.from(model.querySelectorAll("option")).map((o) => o.textContent),
    ).toEqual(["gpt-5.3-codex", "gpt-5.1-codex"]);
  });
});

describe("An optional discriminator is still gated", () => {
  // `AutoPilotTransport | None` serialises as anyOf[{enum}, {type:null}] with no
  // top-level enum. The filter used to require a top-level enum, so making a
  // field optional silently switched the gate off and left the gated option on
  // offer to accounts that cannot use it.
  const optionalTransportSchema = {
    type: "object",
    properties: {
      prompt: { advanced: false, title: "Prompt", type: "string" },
      transport: {
        advanced: false,
        default: null,
        title: "Transport",
        anyOf: [
          {
            enum: ["platform", "codex_app_server"],
            enumNames: ["AutoGPT Platform", "ChatGPT"],
            type: "string",
          },
          { type: "null" },
        ],
      },
      codex_credentials: {
        credentials_provider: ["codex"],
        credentials_types: ["oauth2"],
        discriminator: "transport",
        discriminator_mapping: { codex_app_server: "codex" },
        properties: {
          id: { type: "string" },
          provider: { enum: ["codex"], type: "string" },
          type: { enum: ["oauth2"], type: "string" },
        },
        required: ["id", "provider", "type"],
        title: "Credentials",
        type: "object",
      },
    },
    required: ["prompt"],
  } as RJSFSchema;

  it("removes the gated option from inside anyOf", () => {
    useNodeStore.setState({
      nodes: [
        {
          ...createCodeGenerationNode(),
          data: {
            ...createCodeGenerationNode().data,
            hardcodedValues: { prompt: "hi" },
            inputSchema: optionalTransportSchema,
          },
        },
      ],
      nodeAdvancedStates: {},
    });

    render(
      <CredentialsProvidersContext.Provider value={{}}>
        <FormCreator
          jsonSchema={optionalTransportSchema}
          nodeId="code-generation-node"
          uiType={BlockUIType.STANDARD}
          showHandles={false}
        />
      </CredentialsProvidersContext.Provider>,
    );

    const select = screen.getByLabelText("agpt_%_transport");
    const labels = Array.from(select.querySelectorAll("option")).map(
      (o) => o.textContent,
    );
    expect(labels).toEqual(["AutoGPT Platform"]);
    expect(screen.queryByText("credential")).toBeNull();
  });

  it("clears a hidden credential when switching to platform transport", async () => {
    const node = createCodeGenerationNode();
    useNodeStore.setState({
      nodes: [
        {
          ...node,
          data: {
            ...node.data,
            hardcodedValues: {
              prompt: "hi",
              transport: "codex_app_server",
              codex_credentials: codexCredential,
            },
            inputSchema: optionalTransportSchema,
          },
        },
      ],
      nodeAdvancedStates: {},
    });

    render(
      <CredentialsProvidersContext.Provider
        value={{ codex: makeProvider("codex", "Codex", [codexCredential]) }}
      >
        <FormCreator
          jsonSchema={optionalTransportSchema}
          nodeId="code-generation-node"
          uiType={BlockUIType.STANDARD}
          showHandles={false}
        />
      </CredentialsProvidersContext.Provider>,
    );

    const select = screen.getByLabelText("agpt_%_transport");
    const platform = Array.from(select.querySelectorAll("option")).find(
      (option) => option.textContent === "AutoGPT Platform",
    );
    if (!platform) throw new Error("expected the platform transport option");

    fireEvent.change(select, { target: { value: platform.value } });

    await waitFor(() => {
      const values = useNodeStore
        .getState()
        .getHardCodedValues("code-generation-node");
      expect(values.transport).toBe("platform");
      expect(values).not.toHaveProperty("codex_credentials");
    });
  });
});

describe("LLM blocks keep every model option", () => {
  const llmSchema = {
    type: "object",
    properties: {
      prompt: { advanced: false, title: "Prompt", type: "string" },
      model: {
        advanced: false,
        default: "gpt-4o",
        enum: ["gpt-4o", "claude-opus-4-5-20251101", "llama3.3"],
        title: "Model",
        type: "string",
      },
      credentials: {
        credentials_provider: ["openai", "anthropic", "ollama"],
        credentials_types: ["api_key"],
        discriminator: "model",
        discriminator_mapping: {
          "gpt-4o": "openai",
          "claude-opus-4-5-20251101": "anthropic",
          "llama3.3": "ollama",
        },
        properties: {
          id: { type: "string" },
          provider: {
            enum: ["openai", "anthropic", "ollama"],
            type: "string",
          },
          type: { const: "api_key", type: "string" },
        },
        required: ["id", "provider", "type"],
        title: "Credentials",
        type: "object",
      },
    },
    required: ["prompt", "credentials"],
  } as unknown as RJSFSchema;

  it("keeps all models when every LLM provider is present, as list_providers guarantees", () => {
    useNodeStore.setState({
      nodes: [
        {
          ...createCodeGenerationNode(),
          data: {
            ...createCodeGenerationNode().data,
            hardcodedValues: { prompt: "hi", model: "gpt-4o" },
            inputSchema: llmSchema,
          },
        },
      ],
      nodeAdvancedStates: {},
    });

    render(
      <CredentialsProvidersContext.Provider
        value={{
          openai: makeProvider("openai", "OpenAI", []),
          anthropic: makeProvider("anthropic", "Anthropic", []),
          ollama: makeProvider("ollama", "Ollama", []),
        }}
      >
        <FormCreator
          jsonSchema={llmSchema}
          nodeId="code-generation-node"
          uiType={BlockUIType.STANDARD}
          showHandles={false}
        />
      </CredentialsProvidersContext.Provider>,
    );

    const model = screen.getByLabelText("agpt_%_model");
    expect(
      Array.from(model.querySelectorAll("option")).map((o) => o.textContent),
    ).toEqual(["gpt-4o", "claude-opus-4-5-20251101", "llama3.3"]);
  });
});
