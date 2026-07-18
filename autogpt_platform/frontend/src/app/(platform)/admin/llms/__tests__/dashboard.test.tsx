import { HttpResponse, http } from "msw";
import { describe, expect, it } from "vitest";

import {
  getGetV2AdminListModelsMockHandler200,
  getGetV2AdminListProvidersMockHandler200,
  getGetV2ListCreatorsMockHandler200,
  getGetV2ListMigrationsMockHandler200,
  getGetV2ListRouteWarningsMockHandler200,
  getGetV2ListRoutesMockHandler200,
} from "@/app/api/__generated__/endpoints/admin/admin.msw";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";

import { LlmRegistryDashboard } from "../components/LlmRegistryDashboard/LlmRegistryDashboard";

function modelPayload(overrides: Record<string, unknown> = {}) {
  return {
    id: "model-uuid",
    slug: "moonshotai/kimi-k3",
    display_name: "Kimi K3",
    description: null,
    provider_id: "provider-uuid",
    creator_id: "creator-uuid",
    context_window: 1048576,
    max_output_tokens: 32768,
    price_tier: 3,
    is_enabled: true,
    is_recommended: true,
    kind: "CHAT",
    visibility: "HIDDEN",
    min_subscription_tier: null,
    fallback_model_slug: null,
    source: "LOCAL",
    catalog_removed_at: null,
    supports_tools: true,
    supports_json_output: true,
    supports_reasoning: true,
    supports_parallel_tool_calls: false,
    capabilities: {},
    metadata: {},
    created_at: "2026-07-18T00:00:00Z",
    updated_at: "2026-07-18T00:00:00Z",
    creator: {
      id: "creator-uuid",
      name: "moonshotai",
      display_name: "Moonshot AI",
      description: null,
      website_url: null,
      logo_url: null,
      source: "SEED",
      created_at: null,
      updated_at: null,
    },
    costs: [],
    ...overrides,
  };
}

function mockAllEndpoints() {
  server.use(
    getGetV2AdminListModelsMockHandler200({ models: [modelPayload()] }),
    getGetV2AdminListProvidersMockHandler200({
      providers: [
        {
          id: "provider-uuid",
          name: "open_router",
          display_name: "OpenRouter",
          description: null,
          source: "SEED",
          model_count: 42,
          created_at: null,
          updated_at: null,
        },
      ],
    }),
    getGetV2ListCreatorsMockHandler200({
      creators: [
        {
          id: "creator-uuid",
          name: "moonshotai",
          display_name: "Moonshot AI",
          description: null,
          website_url: "https://moonshot.ai",
          logo_url: null,
          source: "SEED",
          created_at: null,
          updated_at: null,
        },
      ],
    }),
    getGetV2ListMigrationsMockHandler200({
      migrations: [
        {
          id: "mig-uuid",
          source_model_slug: "openai/gpt-old",
          target_model_slug: "openai/gpt-new",
          reason: "Provider retirement",
          node_count: 7,
          custom_credit_cost: null,
          is_reverted: false,
          reverted_at: null,
          created_at: "2026-07-01T00:00:00Z",
        },
      ],
    }),
    getGetV2ListRoutesMockHandler200({
      routes: [
        {
          surface: "copilot",
          mode: "thinking",
          tier: "standard",
          model_slug: "moonshotai/kimi-k3",
          updated_at: null,
        },
      ],
    }),
    getGetV2ListRouteWarningsMockHandler200([
      {
        slug: "moonshotai/kimi-k3s",
        reason: "unknown to the model registry",
        count: 400,
        last_seen: new Date("2026-07-17T18:00:00Z"),
        last_layer: "ld",
      },
    ]),
  );
}

describe("LlmRegistryDashboard", () => {
  it("renders registry models with status, visibility, and source badges", async () => {
    mockAllEndpoints();
    render(<LlmRegistryDashboard />);

    // The slug renders in both the models table and the routing matrix cell.
    expect(
      (await screen.findAllByText("moonshotai/kimi-k3")).length,
    ).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("Kimi K3")).toBeDefined();
    expect(screen.getByText("HIDDEN")).toBeDefined();
    expect(screen.getByText("LOCAL")).toBeDefined();
    expect(screen.getByText("1,048,576")).toBeDefined();
  });

  it("renders the copilot routing matrix with set cells and fall-through cells", async () => {
    mockAllEndpoints();
    render(<LlmRegistryDashboard />);

    const cells = await screen.findAllByText("moonshotai/kimi-k3");
    expect(cells.length).toBeGreaterThanOrEqual(1);
    // Only thinking.standard is set; the other three editable cells fall through.
    expect(screen.getAllByText("— falls through —").length).toBe(3);
  });

  it("surfaces routing resolution warnings with count and reason", async () => {
    mockAllEndpoints();
    render(<LlmRegistryDashboard />);

    expect(await screen.findByText("moonshotai/kimi-k3s")).toBeDefined();
    expect(screen.getByText("unknown to the model registry")).toBeDefined();
    expect(screen.getByText("400")).toBeDefined();
  });

  it("renders providers, creators, and migrations sections", async () => {
    mockAllEndpoints();
    render(<LlmRegistryDashboard />);

    expect(await screen.findByText("OpenRouter")).toBeDefined();
    // Creator display name appears in the models table's creator column too.
    expect(screen.getAllByText("Moonshot AI").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("openai/gpt-old")).toBeDefined();
    expect(screen.getByText("Provider retirement")).toBeDefined();
  });

  it("shows an error card when the models endpoint fails", async () => {
    mockAllEndpoints();
    server.use(
      http.get("*/api/admin/llm/models", () => {
        return new HttpResponse(null, { status: 500 });
      }),
    );
    render(<LlmRegistryDashboard />);

    expect(await screen.findByText("Something went wrong")).toBeDefined();
  });
});
