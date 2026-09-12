import { vi } from "vitest";
import { http, HttpResponse } from "msw";
import type { ProviderMetadata } from "@/app/api/__generated__/models/providerMetadata";
import {
  getGetV1ListCredentialsMockHandler,
  getGetV1ListProvidersMockHandler,
} from "@/app/api/__generated__/endpoints/integrations/integrations.msw";
import { server } from "@/mocks/mock-server";

type MCPServer = NonNullable<ProviderMetadata["mcp_server"]>;

function preset(
  name: string,
  slug: string,
  metadata: Partial<MCPServer>,
): ProviderMetadata {
  return {
    name: `mcp_${slug}`,
    display_name: name,
    description: `${name} account and data tools`,
    supported_auth_types: ["oauth2"],
    mcp_server: {
      server_url: `https://mcp.${slug}.example.com/mcp`,
      documentation_url: `https://docs.example.com/${slug}`,
      setup_instructions: `Choose the permitted access for ${name}.`,
      connection_mode: "hosted",
      auth_mode: "oauth",
      ...metadata,
    },
  };
}

export const authProviders: ProviderMetadata[] = [
  preset("AgentMail", "agentmail", { auth_methods: ["oauth"] }),
  preset("Intercom", "intercom", {
    auth_mode: "token",
    auth_methods: ["bearer"],
  }),
  preset("Langfuse", "langfuse", {
    server_url: null,
    connection_mode: "custom",
    auth_mode: "token",
    auth_methods: ["basic"],
    server_url_options: [
      { label: "EU", url: "https://cloud.langfuse.com/api/public/mcp" },
      { label: "US", url: "https://us.cloud.langfuse.com/api/public/mcp" },
    ],
  }),
  preset("Parallel", "parallel", {
    server_url: "https://search.parallel.ai/mcp",
    oauth_server_url: "https://search.parallel.ai/mcp-oauth",
    auth_mode: "none",
    auth_methods: ["none", "oauth", "bearer"],
  }),
  preset("Customer.io", "customer_io", {
    auth_methods: ["oauth"],
    oauth_scopes: ["read"],
    oauth_write_scopes: ["write"],
    setup_instructions: "Read account data. Allow changes to create drafts.",
  }),
];

export const oauthRequest = vi.fn();
export const discoveryRequest = vi.fn();
export const tokenRequest = vi.fn();

export function setupAuthFixtures() {
  vi.clearAllMocks();
  server.use(
    getGetV1ListCredentialsMockHandler([]),
    getGetV1ListProvidersMockHandler(authProviders),
    http.post("*/api/mcp/oauth/login", async ({ request }) => {
      oauthRequest(await request.json());
      return HttpResponse.json(
        { detail: "Sign-in is unavailable in this test" },
        { status: 400 },
      );
    }),
    http.post("*/api/mcp/discover-tools", async ({ request }) => {
      discoveryRequest(await request.json());
      return HttpResponse.json({ tools: [] });
    }),
    http.post("*/api/mcp/token", async ({ request }) => {
      tokenRequest(await request.json());
      return HttpResponse.json({
        id: "test-credential",
        provider: "mcp",
        type: "oauth2",
        title: "Saved account",
      });
    }),
  );
}
