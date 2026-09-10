import { getGetV2ListLibraryAgentsMockHandler } from "@/app/api/__generated__/endpoints/library/library.msw";
import {
  getGetV2GetSpecificAgentMockHandler,
  getGetV2ListStoreAgentsMockHandler,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import type { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import type { StoreAgentDetails } from "@/app/api/__generated__/models/storeAgentDetails";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { MarketplaceStep } from "./MarketplaceStep";

const storeAgent = {
  slug: "seo-writer",
  agent_name: "SEO Blog Writer",
  agent_image: "",
  creator: "acme",
  creator_avatar: "",
  sub_heading: "Writes optimized blog posts",
  description: "",
  runs: 10,
  rating: 5,
  agent_graph_id: "graph-1",
} as StoreAgent;

function renderMarketplace(
  overrides: Partial<Parameters<typeof MarketplaceStep>[0]> = {},
) {
  const onSubmit = vi.fn();
  const onSkip = vi.fn();
  render(
    <>
      <MarketplaceStep
        color="rose-300"
        submitted={null}
        onSubmit={onSubmit}
        onSkip={onSkip}
        {...overrides}
      />
      <Toaster />
    </>,
  );
  return { onSubmit, onSkip };
}

describe("MarketplaceStep", () => {
  test("shows Skipped when the marketplace step was skipped", () => {
    renderMarketplace({ submitted: [] });
    expect(screen.getByText("Skipped")).toBeDefined();
    expect(
      screen.queryByRole("textbox", {
        name: "Search marketplace and library workflows",
      }),
    ).toBeNull();
  });

  test("skip and empty continue both submit no workflows", async () => {
    const { onSubmit, onSkip } = renderMarketplace();
    await userEvent.click(screen.getByRole("button", { name: "That's it" }));
    expect(onSubmit).toHaveBeenCalledWith([]);

    await userEvent.click(screen.getByRole("button", { name: "Skip" }));
    expect(onSkip).toHaveBeenCalled();
  });

  test("shows only three default marketplace workflows", async () => {
    server.use(
      getGetV2ListStoreAgentsMockHandler({
        agents: [
          { ...storeAgent, agent_name: "Workflow 1", slug: "wf-1" },
          { ...storeAgent, agent_name: "Workflow 2", slug: "wf-2" },
          { ...storeAgent, agent_name: "Workflow 3", slug: "wf-3" },
          { ...storeAgent, agent_name: "Workflow 4", slug: "wf-4" },
        ],
        pagination: {
          total_items: 4,
          total_pages: 1,
          current_page: 1,
          page_size: 3,
        },
      }),
      getGetV2ListLibraryAgentsMockHandler({
        agents: [
          { id: "lib-agent-1", name: "Library Workflow" } as LibraryAgent,
        ],
        pagination: {
          total_items: 1,
          total_pages: 1,
          current_page: 1,
          page_size: 3,
        },
      }),
    );
    renderMarketplace();

    expect(await screen.findByText("Workflow 1")).toBeDefined();
    expect(screen.getByText("Workflow 2")).toBeDefined();
    expect(screen.getByText("Workflow 3")).toBeDefined();
    expect(screen.queryByText("Workflow 4")).toBeNull();
    expect(screen.queryByText("Library Workflow")).toBeNull();
    expect(screen.getAllByRole("button", { name: "Add" })).toHaveLength(3);
  });

  test("searches marketplace and library workflows, then attaches both", async () => {
    server.use(
      getGetV2ListStoreAgentsMockHandler({
        agents: [storeAgent],
        pagination: {
          total_items: 1,
          total_pages: 1,
          current_page: 1,
          page_size: 3,
        },
      }),
      getGetV2ListLibraryAgentsMockHandler({
        agents: [{ id: "lib-agent-1", name: "Local SEO" } as LibraryAgent],
        pagination: {
          total_items: 1,
          total_pages: 1,
          current_page: 1,
          page_size: 3,
        },
      }),
      getGetV2GetSpecificAgentMockHandler({
        store_listing_version_id: "listing-version-42",
        slug: "seo-writer",
        agent_name: "SEO Blog Writer",
        creator: "acme",
      } as StoreAgentDetails),
    );
    const { onSubmit } = renderMarketplace();

    await userEvent.type(
      screen.getByRole("textbox", {
        name: "Search marketplace and library workflows",
      }),
      "seo",
    );

    expect(await screen.findByText("SEO Blog Writer")).toBeDefined();
    expect(await screen.findByText("Local SEO")).toBeDefined();
    expect(screen.queryByText("Marketplace skill")).toBeNull();
    await userEvent.click(screen.getAllByRole("button", { name: "Add" })[0]);
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /Remove SEO Blog Writer/ }),
      ).toBeDefined(),
    );
    await userEvent.click(screen.getByRole("button", { name: "Add" }));
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /Remove Local SEO/ }),
      ).toBeDefined(),
    );
    await userEvent.click(screen.getByRole("button", { name: "That's it" }));

    expect(onSubmit).toHaveBeenCalledWith([
      expect.objectContaining({
        kind: "workflow",
        source: "marketplace",
        id: "listing-version-42",
        name: "SEO Blog Writer",
      }),
      expect.objectContaining({
        kind: "workflow",
        source: "library",
        id: "lib-agent-1",
        name: "Local SEO",
      }),
    ]);
  });
});
