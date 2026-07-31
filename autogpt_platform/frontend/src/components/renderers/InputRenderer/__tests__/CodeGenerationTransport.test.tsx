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

vi.mock("@/components/atoms/Select/Select", () => ({
  Select: ({
    id,
    value,
    onValueChange,
    options,
  }: {
    id: string;
    value?: string;
    onValueChange?: (value: string) => void;
    options: { value: string; label: string }[];
  }) => (
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
  ),
}));

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
