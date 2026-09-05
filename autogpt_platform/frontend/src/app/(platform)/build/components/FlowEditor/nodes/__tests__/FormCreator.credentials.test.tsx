import { BlockUIType } from "@/app/(platform)/build/components/types";
import { FormCreator } from "@/app/(platform)/build/components/FlowEditor/nodes/FormCreator";
import type { CustomNode } from "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/CustomNode";
import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";
import { useHistoryStore } from "@/app/(platform)/build/stores/historyStore";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import {
  CredentialsProvidersContext,
  type CredentialsProvidersContextType,
} from "@/providers/agent-credentials/credentials-provider";
import { render } from "@/tests/integrations/test-utils";
import { screen, waitFor } from "@testing-library/react";
import type { RJSFSchema } from "@rjsf/utils";
import React from "react";
import { beforeEach, describe, expect, it } from "vitest";

// Mirrors AutoPilotBlock.codex_credentials: optional (default null), rendered on
// the block face, and — critically — provider/type are `const` in the subschema
// while `id` is not. RJSF seeds const values into default form state, so this
// field materializes as {provider, type} with no id and no user interaction.
const autopilotSchema = {
  type: "object",
  properties: {
    prompt: { advanced: false, title: "Prompt", type: "string" },
    codex_credentials: {
      advanced: false,
      credentials_provider: ["codex"],
      credentials_types: ["oauth2"],
      default: null,
      properties: {
        id: { title: "Id", type: "string" },
        provider: { const: "codex", title: "Provider", type: "string" },
        type: { const: "oauth2", title: "Type", type: "string" },
      },
      required: ["id", "provider", "type"],
      secret: true,
      title: "ChatGPT / Codex connection",
      type: "object",
    },
  },
  required: ["prompt"],
} as unknown as RJSFSchema;

const requiredUnmappedAutopilotSchema = {
  ...autopilotSchema,
  properties: {
    ...autopilotSchema.properties,
    transport: {
      enum: ["platform", "codex_app_server"],
      title: "Transport",
      type: "string",
    },
    codex_credentials: {
      ...(autopilotSchema.properties?.codex_credentials as RJSFSchema),
      discriminator: "transport",
      discriminator_mapping: { codex_app_server: "codex" },
    },
  },
  required: ["prompt", "codex_credentials"],
} as RJSFSchema;

function makeNode(schema: RJSFSchema): CustomNode {
  return {
    id: "autopilot-node",
    type: "custom",
    position: { x: 0, y: 0 },
    data: {
      hardcodedValues: {},
      title: "AutoPilot",
      description: "Run an autopilot task",
      inputSchema: schema,
      outputSchema: {},
      uiType: BlockUIType.STANDARD,
      block_id: "c069dc6b-c3ed-4c12-b6e5-d47361e64ce6",
      costs: [],
      categories: [],
    },
  } as unknown as CustomNode;
}

function makeProviders(...names: string[]): CredentialsProvidersContextType {
  return makeProvidersWithCredentials(
    Object.fromEntries(names.map((name) => [name, []])),
  );
}

function makeProvidersWithCredentials(
  byProvider: Record<string, unknown[]>,
): CredentialsProvidersContextType {
  const entries = Object.entries(byProvider).map(([name, savedCredentials]) => [
    name,
    {
      provider: name,
      providerName: name,
      savedCredentials,
      isSystemProvider: false,
    },
  ]);
  return Object.fromEntries(entries) as CredentialsProvidersContextType;
}

function renderNode(
  providers: CredentialsProvidersContextType,
  schema: RJSFSchema = autopilotSchema,
  nodeId = "autopilot-node",
) {
  useNodeStore.setState({
    nodes: [{ ...makeNode(schema), id: nodeId } as unknown as CustomNode],
    nodeAdvancedStates: {},
  });
  useEdgeStore.setState({ edges: [] });
  useHistoryStore.setState({ past: [], future: [] });

  return render(
    <CredentialsProvidersContext.Provider value={providers}>
      <FormCreator
        jsonSchema={schema}
        nodeId={nodeId}
        uiType={BlockUIType.STANDARD}
        showHandles={false}
      />
    </CredentialsProvidersContext.Provider>,
  );
}

beforeEach(() => {
  useNodeStore.setState({ nodes: [], nodeAdvancedStates: {} });
});

describe("FormCreator credential emission", () => {
  // Regression: a non-entitled user has no `codex` provider (the backend strips
  // it from /integrations/providers), so this field can never be filled. Before
  // the fix the node still persisted {provider, type} with no id, which reached
  // input_default and made POST /api/graphs return 500 with detail "'id'" from
  // graph_lifecycle_hooks._before_graph_activate.
  it("never persists a credential field that has no id", async () => {
    renderNode({});

    await waitFor(() => {
      expect(
        useNodeStore.getState().getHardCodedValues("autopilot-node"),
      ).toBeDefined();
    });

    const stored = useNodeStore
      .getState()
      .getHardCodedValues("autopilot-node") as Record<string, unknown>;

    expect(stored).not.toHaveProperty("codex_credentials");
  });

  // Seeded as an initial value so the credential travels through FormCreator's
  // handleChange — the same path that strips id-less objects. Writing straight
  // to the store after render would bypass the code under test entirely.
  it("keeps a required credential when its discriminator is unmapped", async () => {
    useNodeStore.setState({
      nodes: [
        {
          ...makeNode(requiredUnmappedAutopilotSchema),
          data: {
            ...makeNode(requiredUnmappedAutopilotSchema).data,
            hardcodedValues: {
              transport: "platform",
              codex_credentials: {
                id: "codex-oauth-1",
                provider: "codex",
                type: "oauth2",
                title: "ChatGPT for Codex",
              },
            },
          },
        } as unknown as CustomNode,
      ],
      nodeAdvancedStates: {},
    });
    useEdgeStore.setState({ edges: [] });
    useHistoryStore.setState({ past: [], future: [] });

    render(
      <CredentialsProvidersContext.Provider
        value={makeProvidersWithCredentials({
          codex: [
            {
              id: "codex-oauth-1",
              provider: "codex",
              type: "oauth2",
              title: "ChatGPT for Codex",
            },
          ],
        })}
      >
        <FormCreator
          jsonSchema={requiredUnmappedAutopilotSchema}
          nodeId="autopilot-node"
          uiType={BlockUIType.STANDARD}
          showHandles={false}
        />
      </CredentialsProvidersContext.Provider>,
    );

    await waitFor(() => {
      expect(
        useNodeStore.getState().getHardCodedValues("autopilot-node"),
      ).toBeDefined();
    });

    const stored = useNodeStore
      .getState()
      .getHardCodedValues("autopilot-node") as Record<string, unknown>;

    expect(stored.codex_credentials).toMatchObject({ id: "codex-oauth-1" });
  });
});

// MCPToolBlock is the other block with a schema-optional rendered credential
// field (default={}), so it exercises the same star path under the literal
// `credentials` name rather than a `*_credentials` one.
const mcpToolSchema = {
  type: "object",
  properties: {
    server_url: { advanced: false, title: "Server Url", type: "string" },
    credentials: {
      advanced: false,
      credentials_provider: ["mcp"],
      credentials_types: ["oauth2"],
      default: {},
      discriminator: "server_url",
      properties: {
        id: { title: "Id", type: "string" },
        provider: { const: "mcp", title: "Provider", type: "string" },
        type: { const: "oauth2", title: "Type", type: "string" },
      },
      required: ["id", "provider", "type"],
      title: "Credentials",
      type: "object",
    },
  },
  required: ["server_url"],
} as unknown as RJSFSchema;

const requiredCodexSchema = {
  type: "object",
  properties: {
    prompt: { advanced: false, title: "Prompt", type: "string" },
    credentials: {
      advanced: false,
      credentials_provider: ["codex"],
      credentials_types: ["oauth2"],
      properties: {
        id: { title: "Id", type: "string" },
        provider: { const: "codex", title: "Provider", type: "string" },
        type: { const: "oauth2", title: "Type", type: "string" },
      },
      required: ["id", "provider", "type"],
      title: "Credentials",
      type: "object",
    },
  },
  required: ["prompt", "credentials"],
} as unknown as RJSFSchema;

describe("unavailable provider", () => {
  // The backend omits providers the user isn't entitled to, so CredentialsInput
  // renders no control. An optional field then has nothing actionable in it.
  it("hides an optional credential row whose provider the user cannot use", async () => {
    renderNode(makeProviders("openai"));

    await waitFor(() => {
      expect(screen.getByText("Prompt")).toBeDefined();
    });

    expect(screen.queryByText("OpenAI credential")).toBeNull();
  });

  it("shows the row once the provider becomes available", async () => {
    renderNode(makeProviders("codex"));

    expect(await screen.findByText("OpenAI credential")).toBeDefined();
  });

  // null context means the fetches haven't settled. Treating that as
  // unavailable would hide the row from entitled users on every page load.
  it("does not hide the row while providers are still loading", async () => {
    renderNode(null as unknown as CredentialsProvidersContextType);

    expect(await screen.findByText("OpenAI credential")).toBeDefined();
  });

  // Found in manual E2E: a PRO user selecting the codex transport on Code
  // Generation saw "Not available on your account." under a field still marked
  // required. The field can't be hidden (the schema requires a credential) but
  // the star demands something the UI offers no way to supply.
  it("does not star a required field whose provider is unavailable", async () => {
    const requiredUnavailableSchema = {
      type: "object",
      properties: {
        prompt: { advanced: false, title: "Prompt", type: "string" },
        credentials: {
          advanced: false,
          credentials_provider: ["codex"],
          credentials_types: ["oauth2"],
          properties: {
            id: { title: "Id", type: "string" },
            provider: { const: "codex", title: "Provider", type: "string" },
            type: { const: "oauth2", title: "Type", type: "string" },
          },
          required: ["id", "provider", "type"],
          title: "Credentials",
          type: "object",
        },
      },
      required: ["prompt", "credentials"],
    } as unknown as RJSFSchema;

    useNodeStore.setState({
      nodes: [
        {
          ...makeNode(requiredUnavailableSchema),
          id: "req-node",
        } as unknown as CustomNode,
      ],
      nodeAdvancedStates: {},
    });
    useEdgeStore.setState({ edges: [] });
    useHistoryStore.setState({ past: [], future: [] });

    render(
      <CredentialsProvidersContext.Provider value={makeProviders("openai")}>
        <FormCreator
          jsonSchema={requiredUnavailableSchema}
          nodeId="req-node"
          uiType={BlockUIType.STANDARD}
          showHandles={false}
        />
      </CredentialsProvidersContext.Provider>,
    );

    const title = await screen.findByText("OpenAI credential");

    expect(title.parentElement?.textContent).not.toContain("*");
  });

  it("still stars a required field whose provider IS available", async () => {
    const requiredAvailableSchema = {
      type: "object",
      properties: {
        credentials: {
          advanced: false,
          credentials_provider: ["github"],
          credentials_types: ["api_key"],
          properties: {
            id: { title: "Id", type: "string" },
            provider: { const: "github", title: "Provider", type: "string" },
            type: { const: "api_key", title: "Type", type: "string" },
          },
          required: ["id", "provider", "type"],
          title: "Credentials",
          type: "object",
        },
      },
      required: ["credentials"],
    } as unknown as RJSFSchema;

    useNodeStore.setState({
      nodes: [
        {
          ...makeNode(requiredAvailableSchema),
          id: "gh-avail",
        } as unknown as CustomNode,
      ],
      nodeAdvancedStates: {},
    });
    useEdgeStore.setState({ edges: [] });
    useHistoryStore.setState({ past: [], future: [] });

    render(
      <CredentialsProvidersContext.Provider value={makeProviders("github")}>
        <FormCreator
          jsonSchema={requiredAvailableSchema}
          nodeId="gh-avail"
          uiType={BlockUIType.STANDARD}
          showHandles={false}
        />
      </CredentialsProvidersContext.Provider>,
    );

    const title = await screen.findByText("Github credential");

    expect(title.parentElement?.textContent).toContain("*");
  });
});

describe("unavailable required field", () => {
  // Regression for the toggle trap: the hide condition must key off the schema's
  // own `required`, not the toggle-adjusted value. Otherwise turning "Optional"
  // on for a required gated field hides the row AND its toggle, with no way back.
  it("keeps a required-but-unavailable row visible when marked optional", async () => {
    useNodeStore.setState({
      nodes: [
        {
          ...makeNode(requiredCodexSchema),
          id: "toggle-node",
          data: {
            ...makeNode(requiredCodexSchema).data,
            metadata: { credentials_optional: true },
          },
        } as unknown as CustomNode,
      ],
      nodeAdvancedStates: {},
    });
    useEdgeStore.setState({ edges: [] });
    useHistoryStore.setState({ past: [], future: [] });

    render(
      <CredentialsProvidersContext.Provider value={makeProviders("openai")}>
        <FormCreator
          jsonSchema={requiredCodexSchema}
          nodeId="toggle-node"
          uiType={BlockUIType.STANDARD}
          showHandles={false}
        />
      </CredentialsProvidersContext.Provider>,
    );

    // The row survives so the toggle stays reachable, but carries no star.
    const title = await screen.findByText("OpenAI credential");
    expect(title.parentElement?.textContent).not.toContain("*");

    // Dropping the star removes the only visual cue that the field matters,
    // so the explanation has to be present to replace it.
    expect(screen.getByText("Not available on your account.")).toBeDefined();
  });

  it("explains why a required gated field cannot be filled", async () => {
    renderNode(makeProviders("openai"), requiredCodexSchema, "required-node");

    const note = await screen.findByText("Not available on your account.");
    // Wired to the control so assistive tech connects the two.
    expect(note.getAttribute("id")).toBeTruthy();
  });
});

describe("credential field required marker", () => {
  // The builder used to derive the marker purely from the node-level
  // credentials_optional toggle, so a schema-optional credential field was
  // still starred. codex_credentials is absent from the schema's `required`
  // array, so it must render without one.
  it("does not star a credential field the schema declares optional", async () => {
    renderNode(makeProviders("codex"));

    const title = await screen.findByText("OpenAI credential");

    expect(title.parentElement?.textContent).not.toContain("*");
  });

  it("still stars a genuinely required field", async () => {
    renderNode(makeProviders("codex"));

    const title = await screen.findByText("Prompt");

    expect(title.parentElement?.textContent).toContain("*");
  });

  it("does not star MCP Tool's schema-optional credentials either", async () => {
    useNodeStore.setState({
      nodes: [
        {
          ...makeNode(mcpToolSchema),
          id: "mcp-node",
        } as unknown as CustomNode,
      ],
      nodeAdvancedStates: {},
    });
    useEdgeStore.setState({ edges: [] });
    useHistoryStore.setState({ past: [], future: [] });

    render(
      <CredentialsProvidersContext.Provider value={makeProviders("mcp")}>
        <FormCreator
          jsonSchema={mcpToolSchema}
          nodeId="mcp-node"
          uiType={BlockUIType.STANDARD}
          showHandles={false}
        />
      </CredentialsProvidersContext.Provider>,
    );

    const title = await screen.findByText("Mcp credential");

    expect(title.parentElement?.textContent).not.toContain("*");
  });

  it("keeps the node-level toggle able to relax a required credential", async () => {
    // The toggle may only relax, never tighten. A schema-required credential
    // with credentials_optional set must lose its star, which is the feature
    // the optional-credentials flag added.
    const requiredCredsSchema = {
      type: "object",
      properties: {
        credentials: {
          advanced: false,
          credentials_provider: ["github"],
          credentials_types: ["api_key"],
          properties: {
            id: { title: "Id", type: "string" },
            provider: { const: "github", title: "Provider", type: "string" },
            type: { const: "api_key", title: "Type", type: "string" },
          },
          required: ["id", "provider", "type"],
          title: "Credentials",
          type: "object",
        },
      },
      required: ["credentials"],
    } as unknown as RJSFSchema;

    useNodeStore.setState({
      nodes: [
        {
          id: "gh-node",
          type: "custom",
          position: { x: 0, y: 0 },
          data: {
            hardcodedValues: {},
            title: "GitHub",
            description: "d",
            inputSchema: requiredCredsSchema,
            outputSchema: {},
            uiType: BlockUIType.STANDARD,
            block_id: "gh",
            costs: [],
            categories: [],
            metadata: { credentials_optional: true },
          },
        } as unknown as CustomNode,
      ],
      nodeAdvancedStates: {},
    });
    useEdgeStore.setState({ edges: [] });
    useHistoryStore.setState({ past: [], future: [] });

    render(
      <CredentialsProvidersContext.Provider value={makeProviders("github")}>
        <FormCreator
          jsonSchema={requiredCredsSchema}
          nodeId="gh-node"
          uiType={BlockUIType.STANDARD}
          showHandles={false}
        />
      </CredentialsProvidersContext.Provider>,
    );

    const title = await screen.findByText("Github credential");

    expect(title.parentElement?.textContent).not.toContain("*");
  });
});
