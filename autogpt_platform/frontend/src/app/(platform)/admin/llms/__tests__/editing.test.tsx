import { HttpResponse, http } from "msw";
import { describe, expect, it, vi } from "vitest";

// Force the large-screen Radix Dialog path — the responsive Drawer (vaul)
// variant doesn't render in jsdom (same mock the Dialog's own tests use).
vi.mock("@/lib/hooks/useBreakpoint", () => ({
  useBreakpoint: () => "lg",
  isLargeScreen: () => true,
}));

import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";

import type { LlmCreatorAdminResponse } from "@/app/api/__generated__/models/llmCreatorAdminResponse";
import type { LlmMigrationAdminResponse } from "@/app/api/__generated__/models/llmMigrationAdminResponse";
import type { LlmModelAdminResponse } from "@/app/api/__generated__/models/llmModelAdminResponse";
import {
  CreatorFormDialog,
  DeleteCreatorDialog,
} from "../components/CreatorDialogs/CreatorDialogs";
import { DeleteModelDialog } from "../components/DeleteModelDialog/DeleteModelDialog";
import { ModelFormDialog } from "../components/ModelFormDialog/ModelFormDialog";
import { RevertMigrationDialog } from "../components/RevertMigrationDialog/RevertMigrationDialog";
import { ToggleModelDialog } from "../components/ToggleModelDialog/ToggleModelDialog";

function model(
  overrides: Partial<LlmModelAdminResponse> = {},
): LlmModelAdminResponse {
  return {
    id: "model-uuid",
    slug: "moonshotai/kimi-k3",
    display_name: "Kimi K3",
    description: null,
    provider_id: "provider-uuid",
    creator_id: null,
    context_window: 1048576,
    max_output_tokens: 32768,
    price_tier: 3,
    is_enabled: true,
    is_recommended: false,
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
    creator: null,
    costs: [],
    ...overrides,
  } as LlmModelAdminResponse;
}

const creator: LlmCreatorAdminResponse = {
  id: "creator-uuid",
  name: "moonshotai",
  display_name: "Moonshot AI",
  description: null,
  website_url: null,
  logo_url: null,
  source: "SEED",
  created_at: null,
  updated_at: null,
} as LlmCreatorAdminResponse;

const migration: LlmMigrationAdminResponse = {
  id: "mig-uuid",
  source_model_slug: "openai/gpt-old",
  target_model_slug: "openai/gpt-new",
  reason: null,
  node_count: 7,
  custom_credit_cost: null,
  is_reverted: false,
  reverted_at: null,
  created_at: "2026-07-01T00:00:00Z",
} as LlmMigrationAdminResponse;

describe("ModelFormDialog", () => {
  it("submits edited display name via PATCH to the model endpoint", async () => {
    let captured: Record<string, unknown> | null = null;
    server.use(
      http.patch("*/api/admin/llm/models/*", async ({ request }) => {
        captured = (await request.json()) as Record<string, unknown>;
        return HttpResponse.json(model());
      }),
    );

    render(
      <ModelFormDialog
        open
        editing={model()}
        providers={[]}
        creators={[creator]}
        models={[model()]}
        onClose={() => undefined}
      />,
    );

    fireEvent.change(screen.getByLabelText("Display name"), {
      target: { value: "Kimi K3 Prime" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }));

    await waitFor(() => expect(captured).not.toBeNull());
    expect(captured!.display_name).toBe("Kimi K3 Prime");
  });

  it("blocks create submit without slug and provider", async () => {
    let posted = false;
    server.use(
      http.post("*/api/admin/llm/models", () => {
        posted = true;
        return HttpResponse.json(model(), { status: 201 });
      }),
    );

    render(
      <ModelFormDialog
        open
        editing={null}
        providers={[]}
        creators={[]}
        models={[]}
        onClose={() => undefined}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Create model" }));

    expect(
      await screen.findByText("Slug and provider are required"),
    ).toBeDefined();
    expect(posted).toBe(false);
  });
});

describe("ToggleModelDialog", () => {
  it("disables the model with is_enabled=false via the toggle endpoint", async () => {
    let captured: Record<string, unknown> | null = null;
    server.use(
      http.post("*/api/admin/llm/models/*/toggle", async ({ request }) => {
        captured = (await request.json()) as Record<string, unknown>;
        return HttpResponse.json({ success: true });
      }),
    );

    render(
      <ToggleModelDialog
        open
        model={model()}
        models={[model(), model({ slug: "openai/gpt-5.2", id: "m2" })]}
        onClose={() => undefined}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Disable model" }));

    await waitFor(() => expect(captured).not.toBeNull());
    expect(captured!.is_enabled).toBe(false);
  });
});

describe("DeleteModelDialog", () => {
  it("deletes the model without replacement when none picked", async () => {
    let calledUrl: string | null = null;
    server.use(
      http.delete("*/api/admin/llm/models/*", ({ request }) => {
        calledUrl = request.url;
        return HttpResponse.json({ success: true });
      }),
    );

    render(
      <DeleteModelDialog
        open
        model={model()}
        models={[model()]}
        onClose={() => undefined}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Delete model" }));

    await waitFor(() => expect(calledUrl).not.toBeNull());
    expect(calledUrl).not.toContain("replacement_model_slug");
  });
});

describe("CreatorFormDialog", () => {
  it("creates a creator with the typed name and display name", async () => {
    let captured: Record<string, unknown> | null = null;
    server.use(
      http.post("*/api/admin/llm/creators", async ({ request }) => {
        captured = (await request.json()) as Record<string, unknown>;
        return HttpResponse.json(creator, { status: 201 });
      }),
    );

    render(<CreatorFormDialog open editing={null} onClose={() => undefined} />);

    fireEvent.change(screen.getByLabelText("Name (slug)"), {
      target: { value: "zai" },
    });
    fireEvent.change(screen.getByLabelText("Display name"), {
      target: { value: "Z.ai" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Create creator" }));

    await waitFor(() => expect(captured).not.toBeNull());
    expect(captured!.name).toBe("zai");
    expect(captured!.display_name).toBe("Z.ai");
  });
});

describe("DeleteCreatorDialog", () => {
  it("deletes the creator on confirm", async () => {
    let called = false;
    server.use(
      http.delete("*/api/admin/llm/creators/*", () => {
        called = true;
        return new HttpResponse(null, { status: 204 });
      }),
    );

    render(
      <DeleteCreatorDialog open creator={creator} onClose={() => undefined} />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Delete creator" }));

    await waitFor(() => expect(called).toBe(true));
  });
});

describe("RevertMigrationDialog", () => {
  it("reverts the migration with re_enable_source_model=true by default", async () => {
    let calledUrl: string | null = null;
    server.use(
      http.post("*/api/admin/llm/migrations/*/revert", ({ request }) => {
        calledUrl = request.url;
        return HttpResponse.json({ success: true });
      }),
    );

    render(
      <RevertMigrationDialog
        open
        migration={migration}
        onClose={() => undefined}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Revert" }));

    await waitFor(() => expect(calledUrl).not.toBeNull());
    expect(calledUrl).toContain("re_enable_source_model=true");
  });
});
